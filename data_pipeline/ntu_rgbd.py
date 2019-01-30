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

"""NTU RGB-D dataset."""


import glob
import os
import numpy as np
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
train_cross_subject = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
train_cross_view = [2, 3]
n_classes = 60


def make_dataset(root, has_skel, evaluation, split, subsample):
  """Returns a list of (video_name, class)."""
  if has_skel or split == 'test' or subsample:
    vid_names = [os.path.splitext(n)[0] \
                 for n in glob.glob1(os.path.join(root, skel_folder_name, 'jjd')
                                     , '*.npy')]
  else:
    vid_names = os.listdir(os.path.join(root, rgb_folder_name))
  vid_names = sorted(vid_names)

  if evaluation == 'cross-subject' and split != 'test':
    vid_names = [n for n in vid_names if int(n[9:12]) in train_cross_subject]
  elif evaluation == 'cross-subject' and split == 'test':
    vid_names = [
        n for n in vid_names if int(n[9:12]) not in train_cross_subject
    ]
  elif evaluation == 'cross-view' and split != 'test':
    vid_names = [n for n in vid_names if int(n[5:8]) in train_cross_view]
  elif evaluation == 'cross-view' and split == 'test':
    vid_names = [n for n in vid_names if int(n[5:8]) not in train_cross_view]
  else:
    raise NotImplementedError

  if subsample:
    labels = np.array([int(n[-3:])-1 for n in vid_names])
    vid_names_subsample = []
    for i in range(n_classes):
      keep = np.where(labels == i)[0]
      keep = keep[np.linspace(0, len(keep)-1, subsample).astype(int)]
      vid_names_subsample += [vid_names[j] for j in keep]
    vid_names = vid_names_subsample

  elif has_skel and split == 'train' and not subsample:
    labels = np.array([int(n[-3:])-1 for n in vid_names])
    vid_names_add = []
    class_54 = np.where(labels == 54)[0].tolist()
    class_58 = np.where(labels == 58)[0].tolist()
    class_59 = np.where(labels == 59)[0].tolist()

    # deterministic oversampling for consistency
    for i in class_54[::7]:
      vid_names_add.append(vid_names[i])
    for i in class_58+class_58[::3]:
      vid_names_add.append(vid_names[i])
    for i in class_59+class_59[::2]+class_59[1::3]:
      vid_names_add.append(vid_names[i])
    vid_names += vid_names_add

  dataset = [(n, int(n[-3:]) - 1) for n in vid_names]

  utils.info('NTU-RGBD: {}, {}, {} videos'.format(split, evaluation,
                                                  len(dataset)))
  return dataset


def rgb_loader(root, vid_name, frame_ids):
  """Loads the RGB data."""
  vid = []
  for frame_ids_s in frame_ids:
    vid_s = []
    for frame_id in frame_ids_s:
      path = os.path.join(root, rgb_folder_name, vid_name,
                          rgb_pattern % (frame_id + 1))
      img = imgproc.imread_rgb('ntu-rgbd', path)
      vid_s.append(img)
    vid.append(vid_s)
  return np.array(vid)


def oflow_loader(root, vid_name, frame_ids):
  """Loads the flow data."""
  vid = []
  for frame_ids_s in frame_ids:
    vid_s = []
    for frame_id in frame_ids_s:
      path = os.path.join(root, oflow_folder_name, vid_name,
                          oflow_pattern % (frame_id + 1))
      img = imgproc.imread_oflow('ntu-rgbd', path)
      vid_s.append(img)
    vid.append(vid_s)
  return np.array(vid)


def depth_loader(root, vid_name, frame_ids):
  """Loads the depth data."""
  vid = []
  for frame_ids_s in frame_ids:
    vid_s = []
    for frame_id in frame_ids_s:
      path = os.path.join(root, depth_folder_name, vid_name,
                          depth_pattern % (frame_id + 1))
      img = imgproc.imread_depth('ntu-rgbd', path)
      vid_s.append(img)
    vid.append(vid_s)
  return np.array(vid)


def jjd_loader(root, vid_name, frame_ids):
  """Loads the skeleton data JJD."""
  path = os.path.join(root, skel_folder_name, 'jjd', vid_name + '.npy')
  skel = np.load(path).astype(np.float32)
  skel = skel[frame_ids]
  return skel


def jjv_loader(root, vid_name, frame_ids):
  """Loads the skeleton data JJV."""
  path = os.path.join(root, skel_folder_name, 'jjv', vid_name + '.npy')
  skel = np.load(path).astype(np.float32)
  skel = skel[frame_ids]
  return skel


def jld_loader(root, vid_name, frame_ids):
  """Loads the skeleton data JLD."""
  path = os.path.join(root, skel_folder_name, 'jld', vid_name + '.npy')
  skel = np.load(path).astype(np.float32)
  skel = skel[frame_ids]
  return skel


class NTU_RGBD(data.Dataset):
  """Class for NTU RGBD Dataset"""
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
               split,
               evaluation,
               modalities,
               n_samples,
               n_frames,
               downsample,
               transforms=None,
               subsample=0):
    """NTU RGBD dataset.

    Args:
      root: dataset root
      split: train to randomly select n_samples samples; test to uniformly
        select n_samples spanning the whole video
      evaluation: one of ['cross-subject', 'cross-view']
      modalities: subset of ['rgb', 'oflow', 'depth', 'jjd', 'jjv', 'jld']
      n_samples: number of samples from the video
      n_frames: number of frames per sample
      downsample: fps /= downsample
      transforms: transformations to apply to data
      subsample: number of samples per class. 0 if using full dataset.
    """
    modalities = utils.unsqueeze(modalities)
    transforms = utils.unsqueeze(transforms)

    # Loader functions
    loaders = {
        'rgb': rgb_loader,
        'oflow': oflow_loader,
        'depth': depth_loader,
        'jjd': jjd_loader,
        'jjv': jjv_loader,
        'jld': jld_loader
    }
    has_skel = any([m in ALL_MODALITIES[3:] for m in modalities])
    dataset = make_dataset(root, has_skel, evaluation, split, subsample)

    self.root = root
    self.split = split
    self.modalities = modalities
    self.n_samples = n_samples
    self.n_frames = n_frames
    self.downsample = downsample
    self.transforms = transforms
    self.loaders = loaders
    self.dataset = dataset

  def __getitem__(self, idx):
    vid_name, label = self.dataset[idx]
    # -1 because len(oflow) = len(rgb)-1
    length = len(
        glob.glob(
            os.path.join(self.root, rgb_folder_name, vid_name,
                         '*.' + rgb_pattern.split('.')[1]))) - 1

    length_ds = length // self.downsample
    if length_ds < self.n_frames:
      frame_ids_s = np.arange(0, length_ds, 1)  # arange: exclusive
      frame_ids_s = np.concatenate(
          (frame_ids_s,
           np.array([frame_ids_s[-1]] * (self.n_frames - length_ds))))
      frame_ids = np.repeat(
          frame_ids_s[np.newaxis, :], self.n_samples,
          axis=0).astype(int) * self.downsample
    else:
      if self.split == 'train':  # randomly select n_samples samples
        starts = np.random.randint(0, length_ds - self.n_frames + 1,
                                   self.n_samples)  # randint: exclusive
      # uniformly select n_samples spanning the whole video
      elif self.split == 'val' or self.split == 'test':
        starts = np.linspace(
            0, length_ds - self.n_frames, self.n_samples,
            dtype=int)  # linspace: inclusive
      else:
        starts = np.arange(0,
                           length_ds - self.n_frames + 1)  # arange: exclusive

      frame_ids = []
      for start in starts:
        frame_ids_s = np.arange(start, start + self.n_frames,
                                1) * self.downsample  # arange: exclusive
        frame_ids.append(frame_ids_s)
      frame_ids = np.stack(frame_ids)

    # load raw data
    inputs = []
    for modality in self.modalities:
      vid = self.loaders[modality](self.root, vid_name, frame_ids)
      inputs.append(vid)

    # transform
    if self.transforms is not None:
      for i in range(len(self.transforms)):
        if self.transforms[i] is not None:
          inputs[i] = self.transforms[i](inputs[i])

    return inputs, label

  def __len__(self):
    return len(self.dataset)
