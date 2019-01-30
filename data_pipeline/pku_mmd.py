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

"""PKU-MMD dataset."""

import glob
import numpy as np
import os
import random
import torch.utils.data as data

import utils
import utils.imgproc as imgproc

rgb_folder_name = 'rgb'
rgb_pattern = 'RGB-%08d.jpg'
oflow_folder_name = 'oflow'
oflow_pattern = 'OFlow-%08d.jpg'
depth_folder_name = 'depth'
depth_pattern = 'Depth-%08d.png'
skel_folder_name = 'skeleton'
feat_folder_name = 'feature'
ALL_MODALITIES = ['rgb', 'oflow', 'depth', 'jjd', 'jjv', 'jld']


def make_dataset(root, evaluation, mode, folder_name=rgb_folder_name, subsample_rate=0):
  """
  mode: train (0), test (1)
  subsample_rate: Rate of subsampling. 0 if using full dataset.
  """
  vid_names = os.listdir(os.path.join(root, folder_name))
  f = open(os.path.join(root, 'split', evaluation + '.txt')).read().split('\n')
  idx = 1 if mode == 0 else 3
  vid_names_split = f[idx].split(', ')[:-1]
  vid_names = sorted(list(set(vid_names).intersection(set(vid_names_split))))

  if mode == 0 and subsample_rate:
    # Subsample for training
    vid_names = vid_names[::subsample_rate]

  dataset = []
  for vid_name in vid_names:
    label = np.loadtxt(
        os.path.join(root, 'label', vid_name + '.txt'),
        delimiter=',',
        dtype=int)
    dataset.append((vid_name, label))

  utils.info('PKU-MMD: {}, {} videos'.format(evaluation, len(dataset)))
  return dataset


def rgb_loader(root, vid_name, frame_ids):
  vid = []
  for frame_ids_s in frame_ids:
    vid_s = []
    for frame_id in frame_ids_s:
      path = os.path.join(root, rgb_folder_name, vid_name,
                          rgb_pattern % (frame_id + 1))
      img = imgproc.imread_rgb('pku-mmd', path)
      vid_s.append(img)
    vid.append(vid_s)
  return np.array(vid)


def oflow_loader(root, vid_name, frame_ids):
  vid = []
  for frame_ids_s in frame_ids:
    vid_s = []
    for frame_id in frame_ids_s:
      path = os.path.join(root, oflow_folder_name, vid_name,
                          oflow_pattern % (frame_id + 1))
      img = imgproc.imread_oflow('pku-mmd', path)
      vid_s.append(img)
    vid.append(vid_s)
  return np.array(vid)


def depth_loader(root, vid_name, frame_ids):
  vid = []
  for frame_ids_s in frame_ids:
    vid_s = []
    for frame_id in frame_ids_s:
      path = os.path.join(root, depth_folder_name, vid_name,
                          depth_pattern % (frame_id + 1))
      img = imgproc.imread_depth('pku-mmd', path)
      vid_s.append(img)
    vid.append(vid_s)
  return np.array(vid)


def jjd_loader(root, vid_name, frame_ids):
  path = os.path.join(root, skel_folder_name, 'jjd', vid_name + '.npy')
  skel = np.load(path).astype(np.float32)
  skel = skel[frame_ids]
  return skel


def jjv_loader(root, vid_name, frame_ids):
  path = os.path.join(root, skel_folder_name, 'jjv', vid_name + '.npy')
  skel = np.load(path).astype(np.float32)
  skel = skel[frame_ids]
  return skel


def jld_loader(root, vid_name, frame_ids):
  path = os.path.join(root, skel_folder_name, 'jld', vid_name + '.npy')
  skel = np.load(path).astype(np.float32)
  skel = skel[frame_ids]
  return skel


def get_overlap(a, b):
  return max(0, min(a[1], b[1]) - max(a[0], b[0]))


class PKU_MMD(data.Dataset):
  MEAN_STD = {
      'rgb': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      'oflow': (0.5, 1. / 255),
      'depth': (4084.1213735 / (255 * 256), 1008.31271366 / (255 * 256)),
      'jjd': (0.53968146, 0.32319776),
      'jjv': (0, 0.35953656),
      'jld': (0.15982792, 0.12776225)
  }

  def __init__(self,
               root,
               mode,
               evaluation,
               modalities,
               step_size,
               n_frames,
               downsample,
               timestep,
               transforms=None,
               subsample_rate=0):
    """PKU_MMD Constructor.

    Contructs the PKU_MMD dataset.

    Args:
      root: dataset root
      mode: train (0), test (1)
      evaluation: one of ['cross-subject', 'cross-view']
      modalities: one of ['rgb', 'oflow', 'depth', 'jjd', 'jjv', 'jld']
      step_size: step size between clips
      n_frames: number of frames per clip
      transforms: transform.
      subsample_rate: sampling rate
      downsample: fps /= downsample
      timestep: number of clips in a sequence.
    """
    modalities = utils.unsqueeze(modalities)
    transforms = utils.unsqueeze(transforms)

    loaders = {
        'rgb': rgb_loader,
        'oflow': oflow_loader,
        'depth': depth_loader,
        'jjd': jjd_loader,
        'jjv': jjv_loader,
        'jld': jld_loader
    }
    self.loaders = loaders
    self.dataset = make_dataset(
        root, evaluation, mode, subsample_rate=subsample_rate)

    self.root = root
    self.modalities = modalities

    self.step_size = step_size
    self.n_frames = n_frames
    self.downsample = downsample
    self.timestep = timestep
    self.all = mode != 0  # True if test mode, return the entire video
    self.transforms = transforms

  def __getitem__(self, idx):
    vid_name, label = self.dataset[idx]
    # label: action_id, start_frame, end_frame, confidence
    # -1 because len(oflow) = len(rgb)-1
    length = len(
        glob.glob(
            os.path.join(self.root, rgb_folder_name, vid_name,
                         '*.' + rgb_pattern.split('.')[1]))) - 1
    length_ds = length // self.downsample

    if self.all:
      # Return entire video
      starts = np.arange(0, length_ds - self.n_frames + 1,
                         self.step_size)  # arange: exclusive
    else:
      start = random.randint(
          0, length_ds - ((self.timestep - 1) * self.step_size + self.n_frames))
      starts = [start + i * self.step_size for i in range(self.timestep)
               ]  # randint: inclusive

    frame_ids = []
    for start in starts:
      frame_ids_s = np.arange(start, start + self.n_frames,
                              1) * self.downsample  # arange: exclusive
      frame_ids.append(frame_ids_s)
    frame_ids = np.stack(frame_ids)

    targets = []
    for frame_ids_s in frame_ids:
      target = 0
      max_ratio = 0.5
      for action_id, start_frame, end_frame, _ in label:
        overlap = get_overlap([frame_ids_s[0], frame_ids_s[-1] - 1],
                              [start_frame, end_frame - 1])
        ratio = overlap / (frame_ids_s[-1] - frame_ids_s[0])
        if ratio > max_ratio:
          target = int(action_id)
      targets.append(target)
    targets = np.stack(targets)

    # load raw data
    inputs = []
    for modality in self.modalities:
      vid = self.loaders[modality](self.root, vid_name, frame_ids)
      inputs.append(vid)

    # transform
    if self.transforms is not None:
      for i, transform in enumerate(self.transforms):
        if transform is not None:
          inputs[i] = transform(inputs[i])

    return inputs, targets, vid_name

  def __len__(self):
    return len(self.dataset)
