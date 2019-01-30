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

"""TODO: One-sentence doc string."""

import glob
import os

import numpy as np
import torch.utils.data as data

import utils
import utils.imgproc as imgproc

rgb_folder_name = 'jpegs_256'
rgb_pattern = 'frame%06d.jpg'
oflow_folder_name = 'tvl1_flow'
oflow_pattern = 'frame%06d.jpg'
train_split_subpath = 'ucfTrainTestlist/trainlist01.txt'
test_split_subpath = 'ucfTrainTestlist/testlist01.txt'
ALL_MODALITIES = ['rgb', 'oflow']


def find_classes(root):
  rgb_folder_path = os.path.join(root, rgb_folder_name)
  classes = [
      n.split('_')[1]
      for n in os.listdir(rgb_folder_path)
      if os.path.isdir(os.path.join(rgb_folder_path, n))
  ]
  classes = sorted(list(set(classes)))
  class_to_idx = {classes[i]: i for i in range(len(classes))}

  return classes, class_to_idx


def make_dataset(root, class_to_idx, train):
  dataset = []
  split_subpath = train_split_subpath if train else test_split_subpath
  split_path = os.path.join(root, split_subpath)

  with open(split_path) as split_file:
    split_lines = split_file.readlines()

    for line in split_lines:
      vid_name = line.split()[0].split('/')[1].replace('.avi', '')
      class_name = vid_name.split('_')[1]
      item = (vid_name, class_to_idx[class_name])
      dataset.append(item)

  return dataset


def rgb_loader(root, vid_name, frame_id):
  rgb_path = os.path.join(root, rgb_folder_name, vid_name,
                          rgb_pattern % frame_id)
  return imgproc.imread_rgb('ucf-101', rgb_path)


def oflow_loader(root, vid_name, frame_id):
  oflow_path_u = os.path.join(root, oflow_folder_name, 'u', vid_name,
                              oflow_pattern % frame_id)
  oflow_path_v = os.path.join(root, oflow_folder_name, 'v', vid_name,
                              oflow_pattern % frame_id)
  return imgproc.imread_oflow('ucf-101', oflow_path_u, oflow_path_v)


class UCF_101(data.Dataset):

  def __init__(self,
               root,
               train,
               modalities,
               n_samples,
               n_frames,
               transforms=None,
               target_transform=None):
    classes, class_to_idx = find_classes(root)
    dataset = make_dataset(root, class_to_idx, train)

    modalities = utils.unsqueeze(modalities)
    transforms = utils.unsqueeze(transforms)

    all_loaders = [rgb_loader, oflow_loader]
    all_modalities = ['rgb', 'oflow']
    loaders = [
        all_loaders[i]
        for i in range(len(all_loaders))
        if all_modalities[i] in modalities
    ]

    assert len(modalities) == len(loaders)

    self.root = root
    self.train = train
    self.modalities = modalities
    self.n_samples = n_samples
    self.n_frames = n_frames
    self.transforms = transforms
    self.target_transform = target_transform

    self.loaders = loaders

    self.classes = classes
    self.class_to_idx = class_to_idx
    self.dataset = dataset

  def __getitem__(self, idx):
    vid_name, target = self.dataset[idx]
    length = len(
        glob.glob(
            os.path.join(self.root, rgb_folder_name, vid_name,
                         '*.' + rgb_pattern.split('.')[1]))) - 1

    if self.train:
      samples = np.random.randint(0, length - self.n_frames + 1,
                                  self.n_samples)  # randint: exclusive
    else:
      if length > self.n_samples:
        samples = np.round(
            np.linspace(0, length - self.n_frames,
                        self.n_samples)).astype(int)  # linspace: inclusive
      else:
        samples = np.arange(0, length - self.n_frames + 1)  # arange: exclusive

    # load raw data
    inputs = []
    for loader in self.loaders:
      vid = []
      for s in samples:
        clip = []
        for t in range(self.n_frames):
          frame_id = s + t + 1
          image = loader(self.root, vid_name, frame_id)
          clip.append(image)
        vid.append(clip)
      inputs.append(np.array(vid))

    # transform
    if self.transforms is not None:
      for i, transform in enumerate(self.transforms):
        if self.transforms[i] is not None:
          inputs[i] = transform(inputs[i])

    return utils.squeeze(inputs), target

  def __len__(self):
    return len(self.dataset)
