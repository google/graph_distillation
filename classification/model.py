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

"""Model to train classification."""

from collections import OrderedDict
import os
from third_party.pytorch.get_cnn import *
from nets.get_distillation_kernel import *
from nets.get_gru import *
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import utils


CNN_MODALITIES = ['rgb', 'oflow', 'depth']
GRU_MODALITIES = ['jjd', 'jjv', 'jld']


class BaseModel:
  """Base class for the model."""

  def __init__(self, modalities, n_classes, n_frames, n_channels, input_sizes,
               hidden_size, n_layers, dropout, lr, lr_decay_rate, ckpt_path):
    super(BaseModel, self).__init__()
    cudnn.benchmark = True

    self.embeds = []
    for _, (modality, n_channels_m, input_size) in enumerate(
        zip(modalities, n_channels, input_sizes)):
      if modality in CNN_MODALITIES:
        self.embeds.append(
            nn.DataParallel(
                get_resnet(n_frames * n_channels_m, n_classes).cuda()))
      elif modality in GRU_MODALITIES:
        self.embeds.append(
            nn.DataParallel(
                get_gru(input_size, hidden_size, n_layers, dropout,
                        n_classes).cuda()))
      else:
        raise NotImplementedError

    self.optimizer = None
    self.criterion_cls = nn.CrossEntropyLoss().cuda()

    self.modalities = modalities
    self.lr = lr
    self.lr_decay_rate = lr_decay_rate
    self.ckpt_path = ckpt_path

  def _forward(self, inputs):
    """Forward pass for all modalities. Return the representation and logits."""
    logits, reprs = [], []

    # Forward pass for all modalities
    for i, (input, embed, modality) in enumerate(
        zip(list(inputs), self.embeds, self.modalities)):
      if modality in CNN_MODALITIES:
        batch_size, n_samples, n_frames, n_channels, h, w = input.size()
        input = input.view(batch_size * n_samples, n_frames * n_channels, h, w)
      elif modality in GRU_MODALITIES:
        batch_size, n_samples, n_frames, input_size = input.size()
        input = input.view(batch_size * n_samples, n_frames, input_size)
      else:
        raise NotImplementedError

      logit, representation = embed(input)
      logit = logit.view(batch_size, n_samples, -1)
      representation = representation.view(batch_size, n_samples, -1)

      logits.append(logit.mean(1))
      reprs.append(representation.mean(1))

    logits = torch.stack(logits)
    reprs = torch.stack(reprs)
    return [logits, reprs]

  def _backward(self, results, label):
    raise NotImplementedError

  def train(self, inputs, label):
    """Train the model.

    Args:
      inputs: a list, each is batch_size x n_sample x n_frames x
      (n_channels x h x w) or (input_size).
      label: batch_size x n_samples.
    Returns:
      info: dictionary of results
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

  def test(self, inputs, label):
    """Test the model.

    Args:
      inputs: a list, each is batch_size x n_sample x n_frames x
        (n_channels x h x w) or (input_size).
      label: batch_size x n_samples.
    Returns:
      info_acc: dictionary of results
    """
    for embed in self.embeds:
      embed.eval()

    inputs = [Variable(x.cuda(), volatile=True) for x in inputs]
    label = Variable(label.cuda(), volatile=True)

    logits, _ = self._forward(inputs)
    info_acc = self._get_acc(logits, label)

    return OrderedDict(info_acc), logits

  def _get_acc(self, logits, label):
    info_acc = []
    for _, (logit, modality) in enumerate(zip(logits, self.modalities)):
      acc, _, label = utils.get_stats(logit, label)
      info_acc.append(('acc_{}'.format(modality), acc))
    return info_acc

  def lr_decay(self):
    lrs = []
    for param_group in self.optimizer.param_groups:
      param_group['lr'] *= self.lr_decay_rate
      lrs.append(param_group['lr'])
    return lrs

  def save(self, epoch):
    path = os.path.join(self.ckpt_path, 'embed_{}.pth'.format(epoch))
    torch.save(self.embeds[self.to_idx].state_dict(), path)

  def load(self, load_ckpt_paths, epoch=200):
    """Load trained models."""
    assert len(load_ckpt_paths) == len(self.embeds)
    for i, ckpt_path in enumerate(load_ckpt_paths):
      if len(ckpt_path) > 0:
        path = os.path.join(ckpt_path, 'embed_{}.pth'.format(epoch))
        self.embeds[i].load_state_dict(torch.load(path))
        utils.info('{}: ckpt {} loaded'.format(self.modalities[i], path))
      else:
        utils.info('{}: training from scratch'.format(self.modalities[i]))


class SingleStream(BaseModel):
  """Model to train a single modality."""

  def __init__(self, *args, **kwargs):
    super(SingleStream, self).__init__(*args, **kwargs)
    assert len(self.embeds) == 1
    self.optimizer = optim.SGD(
        self.embeds[0].parameters(),
        lr=self.lr,
        momentum=0.9,
        weight_decay=5e-4)
    self.to_idx = 0

  def _backward(self, results, label):
    logits, _ = results
    logits = logits.view(*logits.size()[1:])
    loss = self.criterion_cls(logits, label)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    info_loss = [('loss', loss.data[0])]
    return info_loss


class GraphDistillation(BaseModel):
  """Model to train with graph distillation.

  xfer_to is the modality to train.
  """

  def __init__(self, modalities, n_classes, n_frames, n_channels, input_sizes,
               hidden_size, n_layers, dropout, lr, lr_decay_rate, ckpt_path,
               w_losses, w_modalities, metric, xfer_to, gd_size, gd_reg):
    super(GraphDistillation, self).__init__( \
               modalities, n_classes, n_frames, n_channels, input_sizes,
               hidden_size, n_layers, dropout, lr, lr_decay_rate, ckpt_path)

    # Index of the modality to distill
    to_idx = self.modalities.index(xfer_to)
    from_idx = [x for x in range(len(self.modalities)) if x != to_idx]
    assert len(from_idx) >= 1

    # Prior
    w_modalities = [w_modalities[i] for i in from_idx
                   ]  # remove modality being transferred to
    gd_prior = utils.softmax(w_modalities, 0.25)
    # Distillation model
    self.distillation_kernel = get_distillation_kernel(
        n_classes, hidden_size, gd_size, to_idx, from_idx, gd_prior, gd_reg,
        w_losses, metric).cuda()

    params = list(self.embeds[to_idx].parameters()) + \
             list(self.distillation_kernel.parameters())
    self.optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    self.xfer_to = xfer_to
    self.to_idx = to_idx
    self.from_idx = from_idx

  def _forward(self, inputs):
    logits, reprs = super(GraphDistillation, self)._forward(inputs)
    # Get edge weights of the graph
    graph = self.distillation_kernel(logits, reprs)
    return logits, reprs, graph

  def _backward(self, results, label):
    logits, reprs, graph = results  # graph: size = len(from_idx) x batch_size
    info_loss = []

    # Classification loss
    loss_cls = self.criterion_cls(logits[self.to_idx], label)
    # Graph distillation loss
    loss_reg, loss_logit, loss_repr = \
        self.distillation_kernel.distillation_loss(logits, reprs, graph)

    loss = loss_cls + loss_reg + loss_logit + loss_repr
    loss.backward()
    if self.xfer_to in GRU_MODALITIES:
      torch.nn.utils.clip_grad_norm(self.embeds[self.to_idx].parameters(), 5)
    self.optimizer.step()
    self.optimizer.zero_grad()

    info_loss = [('loss_cls', loss_cls.data[0]), ('loss_reg', loss_reg.data[0]),
                 ('loss_logit', loss_logit.data[0]), ('loss_repr',
                                                      loss_repr.data[0])]
    return info_loss
