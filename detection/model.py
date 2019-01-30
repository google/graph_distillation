# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model to train detection."""

from collections import OrderedDict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

import utils
from nets.get_tad import *
from nets.get_distillation_kernel import *

CNN_MODALITIES = ['rgb', 'oflow', 'depth']
GRU_MODALITIES = ['jjd', 'jjv', 'jld']


class BaseModel:
  def __init__(self, modalities, n_classes, n_frames, n_channels, input_sizes,
               hidden_size, n_layers, dropout, hidden_size_seq, n_layers_seq,
               dropout_seq, bg_w, lr, lr_decay_rate, to_idx, ckpt_path):
    super(BaseModel, self).__init__()
    cudnn.benchmark = True
    utils.info('{} modality'.format(modalities[to_idx]))

    self.embeds = []
    for i, m in enumerate(modalities):
      encoder_type = 'cnn' if m in CNN_MODALITIES else 'rnn'
      embed = nn.DataParallel(
          get_tad(n_classes, n_frames, n_channels[i], input_sizes[i],
                  hidden_size, n_layers, dropout, hidden_size_seq, n_layers_seq,
                  dropout_seq, encoder_type).cuda())
      self.embeds.append(embed)

    # Multiple optimizers
    self.optimizers = []
    self.lr_decay_rates = []
    # Visual encoder: SGD
    visual_params = list(self.embeds[to_idx].module.embed.parameters())
    visual_optimizer = optim.SGD(
        visual_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    self.optimizers.append(visual_optimizer)
    self.lr_decay_rates.append(lr_decay_rate)
    # Sequence encoder: Adam
    sequence_params = list(self.embeds[to_idx].module.rnn.parameters()) + \
                      list(self.embeds[to_idx].module.fc.parameters())
    sequence_optimizer = optim.Adam(sequence_params, lr=1e-3)
    self.optimizers.append(sequence_optimizer)
    self.lr_decay_rates.append(1)  # No learning rate decay for Adam

    # Weighted cross-entropy loss
    self.criterion_cls = nn.CrossEntropyLoss(
        torch.FloatTensor([bg_w] + [1] * n_classes)).cuda()

    self.n_classes = n_classes
    self.modalities = modalities
    self.to_idx = to_idx
    self.ckpt_path = ckpt_path

  def _forward(self, inputs):
    """Forward pass for all modalities.
    """
    logits, reprs = [], []
    for i in range(len(inputs)):
      logit, repr = self.embeds[i](inputs[i])
      logits.append(logit)
      reprs.append(repr)

    logits = torch.stack(logits)
    reprs = torch.stack(reprs)
    return [logits, reprs]

  def _backward(self, results, label):
    raise NotImplementedError

  def train(self, inputs, label):
    """Train model.
    :param inputs: a list, each is batch_size x timestep x n_frames x (n_channels x h x w) or (input_size)
    :param label: batch_size x timestep
    """
    for embed in self.embeds:
      embed.train()

    for i in range(len(inputs)):
      inputs[i] = Variable(inputs[i].cuda(), requires_grad=False)
    label = Variable(label.cuda(), requires_grad=False)

    results = self._forward(inputs)
    info_loss = self._backward(results, label)
    info_acc = self._get_acc(results[0], label)
    return OrderedDict(info_loss + info_acc)

  def test(self, inputs, label, timestep):
    '''Test model.
    param timestep: split into segments of length timestep.
    '''
    for embed in self.embeds:
      embed.eval()

    input = Variable(inputs[0].cuda(), requires_grad=False)
    label = Variable(label.cuda(), requires_grad=False)
    length = input.size(1)

    # Split video into segments
    input, start_indices = utils.get_segments(input, timestep)
    inputs = [input]

    logits, _ = self._forward(inputs)
    logits = utils.to_numpy(logits).squeeze(0)
    all_logits = [[] for i in range(length)]
    for i in range(len(start_indices)):
      s = start_indices[i]
      for j in range(timestep):
        all_logits[s + j].append(logits[i][j])
    # Average logits for each time step.
    final_logits = np.zeros((length, self.n_classes + 1))
    for i in range(length):
      final_logits[i] = np.mean(all_logits[i], axis=0)
    logits = final_logits

    info_acc = self._get_acc([torch.Tensor(logits)], label)
    scores = utils.softmax(logits, axis=1)
    return OrderedDict(info_acc), logits, scores

  def _get_acc(self, logits, label):
    """Get detection statistics for modality.
    """
    info_acc = []
    for i, m in enumerate(self.modalities):
      logit = logits[i].view(-1, self.n_classes + 1)
      label = label.view(-1)
      stats = utils.get_stats_detection(logit, label, self.n_classes + 1)
      info_acc.append(('ap_{}'.format(m), stats[0]))
      info_acc.append(('acc_{}'.format(m), stats[1]))
      info_acc.append(('acc_bg_{}'.format(m), stats[2]))
      info_acc.append(('acc_action_{}'.format(m), stats[3]))
    return info_acc

  def save(self, epoch):
    path = os.path.join(self.ckpt_path, 'embed_{}.pth'.format(epoch))
    torch.save(self.embeds[self.to_idx].state_dict(), path)

  def load(self, load_ckpt_paths, options, epoch=200):
    """Load checkpoints.
    """
    assert len(load_ckpt_paths) == len(self.embeds)
    for i in range(len(self.embeds)):
      ckpt_path = load_ckpt_paths[i]
      load_opt = options[i]
      if len(ckpt_path) == 0:
        utils.info('{}: training from scratch'.format(self.modalities[i]))
        continue

      if load_opt == 0:  # load teacher model (visual + sequence)
        path = os.path.join(ckpt_path, 'embed_{}.pth'.format(epoch))
        ckpt = torch.load(path)
        try:
          self.embeds[i].load_state_dict(ckpt)
        except:
          utils.warn('Check that the "modalities" argument is correct.')
          exit(0)
        utils.info('{}: ckpt {} loaded'.format(self.modalities[i], path))
      elif load_opt == 1:  # load pretrained visual encoder
        ckpt = torch.load(ckpt_path)
        # Change keys in the ckpt
        new_state_dict = OrderedDict()
        for key in list(ckpt.keys())[:-2]:  # exclude fc weights
          new_key = key[7:]  # Remove 'module.'
          new_state_dict[new_key] = ckpt[key]
        # update state_dict
        state_dict = self.embeds[i].module.embed.state_dict()
        state_dict.update(new_state_dict)
        self.embeds[i].module.embed.load_state_dict(state_dict)
        utils.info('{}: visual encoder from {} loaded'.format(
            self.modalities[i], ckpt_path))
      else:
        raise NotImplementedError

  def lr_decay(self):
    lrs = []
    for optimizer, decay_rate in zip(self.optimizers, self.lr_decay_rates):
      for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate
        lrs.append(param_group['lr'])
    return lrs


class SingleStream(BaseModel):
  """Model to train a single modality.
  """

  def __init__(self, *args, **kwargs):
    super(SingleStream, self).__init__(*args, **kwargs)
    assert len(self.embeds) == 1

  def _backward(self, results, label):
    logits, _ = results
    logits = logits.view(-1, logits.size(-1))

    loss = self.criterion_cls(logits, label.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm(self.embeds[self.to_idx].parameters(), 5)
    for optimizer in self.optimizers:
      optimizer.step()
      optimizer.zero_grad()

    info_loss = [('loss', loss.data[0])]
    return info_loss


class GraphDistillation(BaseModel):
  """Model to train with graph distillation.

  xfer_to is the modality to train.
  """

  def __init__(self, modalities, n_classes, n_frames, n_channels, input_sizes,
               hidden_size, n_layers, dropout, hidden_size_seq, n_layers_seq,
               dropout_seq, bg_w, lr, lr_decay_rate, to_idx, ckpt_path,
               w_losses, w_modalities, metric, xfer_to, gd_size, gd_reg):
    super(GraphDistillation, self).__init__(\
               modalities, n_classes, n_frames, n_channels, input_sizes,
               hidden_size, n_layers, dropout, hidden_size_seq, n_layers_seq, dropout_seq,
               bg_w, lr, lr_decay_rate, to_idx, ckpt_path)

    # Index of the modality to distill
    to_idx = self.modalities.index(xfer_to)
    from_idx = [x for x in range(len(self.modalities)) if x != to_idx]
    assert len(from_idx) >= 1

    # Prior
    w_modalities = [w_modalities[i] for i in from_idx
                   ]  # remove modality being transferred to
    gd_prior = utils.softmax(w_modalities, 0.25)
    # Distillation model
    self.distillation_kernel = \
        get_distillation_kernel(n_classes + 1, hidden_size, gd_size, to_idx, from_idx,
                                gd_prior, gd_reg, w_losses, metric).cuda()

    # Add optimizer to self.optimizers
    gd_optimizer = optim.SGD(
        self.distillation_kernel.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4)
    self.optimizers.append(gd_optimizer)
    self.lr_decay_rates.append(lr_decay_rate)

    self.xfer_to = xfer_to
    self.to_idx = to_idx
    self.from_idx = from_idx

  def _forward(self, inputs):
    logits, reprs = super(GraphDistillation, self)._forward(inputs)
    n_modalities, batch_size, length, _ = logits.size()
    logits = logits.view(n_modalities, batch_size * length, -1)
    reprs = reprs.view(n_modalities, batch_size * length, -1)
    # Get edge weights of the graph
    graph = self.distillation_kernel(logits, reprs)
    return logits, reprs, graph

  def _backward(self, results, label):
    logits, reprs, graph = results  # graph: size (len(from_idx) x batch_size)
    label = label.view(-1)
    info_loss = []

    # Classification loss
    loss_cls = self.criterion_cls(logits[self.to_idx], label)
    # Graph distillation loss
    loss_reg, loss_logit, loss_repr = \
        self.distillation_kernel.distillation_loss(logits, reprs, graph)

    loss = loss_cls + loss_reg + loss_logit + loss_repr
    loss.backward()
    torch.nn.utils.clip_grad_norm(self.embeds[self.to_idx].parameters(), 5)
    for optimizer in self.optimizers:
      optimizer.step()
      optimizer.zero_grad()

    info_loss = [('loss_cls', loss_cls.data[0]), ('loss_reg', loss_reg.data[0]),
                 ('loss_logit', loss_logit.data[0]), ('loss_repr',
                                                      loss_repr.data[0])]
    return info_loss
