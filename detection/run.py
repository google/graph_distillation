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

"""Train detection"""

import argparse
import os

import utils
import utils.logging as logging
from .get_dataloader import *
from .get_model import *
from .evaluation.map import calc_map

ALL_MODALITIES = ['rgb', 'oflow', 'depth', 'jjd', 'jjv', 'jld']

parser = argparse.ArgumentParser()

# experimental settings
parser.add_argument('--n_workers', type=int, default=24)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'])

# ckpt and logging
parser.add_argument('--ckpt_path', type=str, default='./ckpt',
                    help='directory path that stores all checkpoints')
parser.add_argument('--ckpt_name', type=str, default='ckpt')
parser.add_argument('--pretrained_ckpt_name', type=str, default='ckpt',
                    help='name of the teacher detection models')
parser.add_argument('--load_epoch', type=int, default=400,
                    help='epoch to load teacher model')
parser.add_argument('--visual_encoder_ckpt_path', type=str, default='',
                    help='classification checkpoint to initialize'
                    'the visual encoder weights')
parser.add_argument('--load_ckpt_path', type=str, default='',
                    help='checkpoint path to load for testing')
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--save_every', type=int, default=50)

# hyperparameters
parser.add_argument('--batch_sizes', type=int, nargs='+', default=[8, 1],
                    help='batch sizes: [train, test]')
parser.add_argument('--n_epochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay_at', type=int, nargs='+', default=[250, 350])
parser.add_argument('--lr_decay_rate', type=float, default=0.1)

# data pipeline
parser.add_argument('--dset', type=str, default='pku-mmd')
parser.add_argument('--dset_path', type=str,
                    default=os.path.join(os.environ['HOME'], 'slowbro'))
parser.add_argument('--modalities', type=str, nargs='+',
                    choices=['rgb', 'oflow', 'depth', 'jjd', 'jjv', 'jld'])
parser.add_argument('--step_size', type=int, default=10,
                    help='step size between samples (after downsample)')
parser.add_argument('--n_frames', type=int, default=10,
                    help='num frames per clip')
parser.add_argument('--downsample', type=int, default=3,
                    help='fps /= downsample')
parser.add_argument('--timestep', type=int, default=10,
                    help='number of clips in a sequence')
parser.add_argument('--bg_w', type=float, default=0.5)
parser.add_argument('--subsample_rate', type=int, default=20,
                    help='rate to subsample the dataset. 0: False (use full dataset)')

# visual encoder GRU (for modalities jjd, jjv, jld)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--n_layers', type=int, default=3)
# sequence encoder GRU
parser.add_argument('--dropout_seq', type=float, default=0.5)
parser.add_argument('--hidden_size_seq', type=int, default=512)
parser.add_argument('--n_layers_seq', type=int, default=1)

# Graph Distillation parameters
parser.add_argument('--metric', type=str, default='cosine',
                    choices=['cosine', 'kl', 'l2', 'l1'],
                    help='distance metric for distillation loss')
parser.add_argument('--w_losses', type=float, nargs='+', default=[10, 1],
                    help='weights for losses: [logit, repr]')
parser.add_argument('--w_modalities', type=float, nargs='+',
                    default=[1, 1, 1, 1, 1, 1],
                    help='modality prior')
parser.add_argument('--xfer_to', type=str, default='',
                    help='modality to train with graph distillation')
parser.add_argument('--gd_size', type=int, default=32,
                    help='hidden size of graph distillation')
parser.add_argument('--gd_reg', type=float, default=10,
                    help='regularization for graph distillation')


def single_stream(opt):
  """Train a single modality from scratch."""
  # Checkpoint path example: ckpt_path/pku-mmd/rgb/ckpt
  opt.ckpt_path = os.path.join(opt.ckpt_path, opt.dset,
                               opt.modalities[0], opt.ckpt_name)
  os.makedirs(opt.ckpt_path, exist_ok=True)
  if opt.split == 'train':
    if opt.visual_encoder_ckpt_path != '':
      assert os.path.exists(opt.visual_encoder_ckpt_path), \
             '{} does not exist'.format(opt.visual_encoder_ckpt_path)
    opt.load_ckpt_paths = [opt.visual_encoder_ckpt_path]
    opt.load_opts = [1] # load visual encoder
  else:
    opt.load_ckpt_paths = [opt.load_ckpt_path]
    opt.load_opts = [0] # load visual + sequence encoder

  # Data loader and model
  dataloader = get_dataloader(opt)
  model = get_model(opt)
  if opt.split == 'train':
    train(opt, model, dataloader)
  else:
    test(opt, model, dataloader)


def multi_stream(opt):
  """Train a modality with graph distillation from other modalities."""
  assert opt.xfer_to in opt.modalities, 'xfer_to must be in opt.modalities'
  # Checkpoints to load
  opt.load_ckpt_paths = []
  opt.load_opts = []
  for m in opt.modalities:
    if m != opt.xfer_to:
      # Checkpoint from single_stream
      path = os.path.join(opt.ckpt_path, opt.dset, m, opt.pretrained_ckpt_name)
      assert os.path.exists(path), '{} checkpoint does not exist.'.format(path)
      opt.load_ckpt_paths.append(path)
      opt.load_opts.append(0)
    else:
      opt.load_ckpt_paths.append(opt.visual_encoder_ckpt_path)
      opt.load_opts.append(1)

  # Checkpoint path example: ckpt_path/ntu-rgbd/xfer_rgb/ckpt_rgb_depth
  opt.ckpt_path = os.path.join(
      opt.ckpt_path, opt.dset, 'xfer_{}'.format(opt.xfer_to), '{}_{}'.format(
          opt.ckpt_name, '_'.join([m for m in opt.modalities])))
  os.makedirs(opt.ckpt_path, exist_ok=True)

  # Data loader and model
  dataloader = get_dataloader(opt)
  model = get_model(opt)
  train(opt, model, dataloader)


def train(opt, model, dataloader):
  # Logging
  logger = logging.Logger(opt.ckpt_path, opt.split)
  stats = logging.Statistics(opt.ckpt_path, opt.split)
  logger.log(opt)

  model.load(opt.load_ckpt_paths, opt.load_opts, opt.load_epoch)
  for epoch in range(1, opt.n_epochs + 1):
    for step, data in enumerate(dataloader, 1):
      # inputs is a list of input of each modality
      inputs, label, _ = data
      ret = model.train(inputs, label)
      update = stats.update(len(label), ret)
      if utils.is_due(step, opt.print_every):
        utils.info('epoch {}/{}, step {}/{}: {}'.format(
            epoch, opt.n_epochs, step, len(dataloader), update))

    logger.log('[Summary] epoch {}/{}: {}'.format(epoch, opt.n_epochs,
                                                  stats.summarize()))

    if utils.is_due(epoch, opt.n_epochs, opt.save_every):
      model.save(epoch)
      stats.save()
      logger.log('***** saved *****')

    if utils.is_due(epoch, opt.lr_decay_at):
      lrs = model.lr_decay()
      logger.log('***** lr decay *****: {}'.format(lrs))


def test(opt, model, dataloader):
  # Logging
  logger = logging.Logger(opt.ckpt_path, opt.split)
  stats = logging.Statistics(opt.ckpt_path, opt.split)
  logger.log(opt)

  model.load(opt.load_ckpt_paths, opt.load_opts, opt.load_epoch)
  all_scores = []
  video_names = []
  for step, data in enumerate(dataloader, 1):
    inputs, label, vid_name = data
    info_acc, logits, scores = model.test(inputs, label, opt.timestep)

    all_scores.append(scores)
    video_names.append(vid_name[0])
    update = stats.update(logits.shape[0], info_acc)
    if utils.is_due(step, opt.print_every):
      utils.info('step {}/{}: {}'.format(step, len(dataloader), update))

  logger.log('[Summary] {}'.format(stats.summarize()))

  # Evaluate
  iou_thresholds = [0.1, 0.3, 0.5]
  groundtruth_dir = os.path.join(opt.dset_path, opt.dset, 'groundtruth',
                                 'validation/cross-subject')
  assert os.path.exists(groundtruth_dir), '{} does not exist'.format(groundtruth_dir)
  mean_aps = calc_map(opt, all_scores, video_names, groundtruth_dir, iou_thresholds)

  for i in range(len(iou_thresholds)):
    logger.log('IoU: {}, mAP: {}'.format(iou_thresholds[i], mean_aps[i]))


if __name__ == '__main__':
  opt = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

  if opt.split == 'test':
    assert len(opt.modalities) == 1, 'specify only 1 modality for testing'
    assert len(opt.load_ckpt_path) > 0, 'specify load_ckpt_path for testing'

  if len(opt.modalities) == 1:
    single_stream(opt)
  else:
    multi_stream(opt)
