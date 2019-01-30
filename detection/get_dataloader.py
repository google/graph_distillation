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

"""Get data loader for detection."""

import os
import torch.utils.data as data
import torchvision.transforms as transforms

import third_party.two_stream_pytorch.video_transforms as vtransforms
from data_pipeline.pku_mmd import PKU_MMD

CNN_MODALITIES = ['rgb', 'oflow', 'depth']
GRU_MODALITIES = ['jjd', 'jjv', 'jld']


def get_dataloader(opt):
  idx_t = 0 if opt.split == 'train' else 1

  xforms = []
  for modality in opt.modalities:
    if opt.dset == 'pku-mmd':
      mean, std = PKU_MMD.MEAN_STD[modality]
    else:
      raise NotImplementedError

    if opt.split == 'train' and (modality == 'rgb' or modality == 'depth'):
      xform = transforms.Compose([
          vtransforms.RandomSizedCrop(224),
          vtransforms.RandomHorizontalFlip(),
          vtransforms.ToTensor(),
          vtransforms.Normalize(mean, std)
      ])
    elif opt.split == 'train' and modality == 'oflow':
      # Special handling when flipping optical flow
      xform = transforms.Compose([
          vtransforms.RandomSizedCrop(224, True),
          vtransforms.RandomHorizontalFlip(True),
          vtransforms.ToTensor(),
          vtransforms.Normalize(mean, std)
      ])
    elif opt.split != 'train' and modality in CNN_MODALITIES:
      xform = transforms.Compose([
          vtransforms.Scale(256),
          vtransforms.CenterCrop(224),
          vtransforms.ToTensor(),
          vtransforms.Normalize(mean, std)
      ])
    elif modality in GRU_MODALITIES:
      xform = transforms.Compose([vtransforms.SkelNormalize(mean, std)])
    else:
      raise Exception

    xforms.append(xform)

  if opt.dset == 'pku-mmd':
    root = os.path.join(opt.dset_path, opt.dset)
    dset = PKU_MMD(root, idx_t, 'cross-subject', opt.modalities, opt.step_size,
                   opt.n_frames, opt.downsample, opt.timestep, xforms,
                   opt.subsample_rate)
  else:
    raise NotImplementedError

  dataloader = data.DataLoader(
      dset,
      batch_size=opt.batch_sizes[idx_t],
      shuffle=(opt.split == 'train'),
      num_workers=opt.n_workers)

  return dataloader
