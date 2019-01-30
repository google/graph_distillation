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

"""Utility functions for reading and processing data."""

import cv2
import numpy as np
from scipy import interpolate
from scipy.misc import imresize


def imread_rgb(dset, path):
  if dset == 'ucf-101':
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return rgb[:, :-1]  # oflow is 1px smaller than rgb in ucf-101
  elif dset == 'ntu-rgbd' or dset == 'pku-mmd' or dset == 'cad-60':
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return rgb
  else:
    assert False


def imread_oflow(dset, *paths):
  if dset == 'ucf-101':
    path_u, path_v = paths
    oflow_u = cv2.imread(path_u, cv2.IMREAD_GRAYSCALE)
    oflow_v = cv2.imread(path_v, cv2.IMREAD_GRAYSCALE)
    oflow = np.stack((oflow_u, oflow_v), axis=2)
    return oflow
  elif dset == 'ntu-rgbd' or dset == 'pku-mmd' or dset == 'cad-60':
    path = paths[0]
    oflow = cv2.imread(path)[..., ::-1][..., :2]
    return oflow
  else:
    assert False


def imread_depth(dset, path):
  # dset == 'ntu-rgbd' or dset == 'pku-mmd'
  depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, np.newaxis]
  depth = np.clip(depth/256, 0, 255).astype(np.uint8)
  return depth


def inpaint(img, threshold=1):
  h, w = img.shape[:2]

  if len(img.shape) == 3:  # RGB
    mask = np.all(img == 0, axis=2).astype(np.uint8)
    img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

  else:  # depth
    mask = np.where(img > threshold)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
    img = np.ravel(img[mask])
    interp = interpolate.NearestNDInterpolator(xym, img)
    img = interp(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

  return img


def resize(video, size, interpolation):
  """
  :param video: ... x h x w x num_channels
  :param size: (h, w)
  :param interpolation: 'bilinear', 'nearest'
  :return:
  """
  shape = video.shape[:-3]
  num_channels = video.shape[-1]
  video = video.reshape((-1, *video.shape[-3:]))
  resized_video = np.zeros((video.shape[0], *size, video.shape[-1]))

  for i in range(video.shape[0]):
    if num_channels == 3:
      resized_video[i] = imresize(video[i], size, interpolation)
    elif num_channels == 2:
      resized_video[i, ..., 0] = imresize(video[i, ..., 0], size, interpolation)
      resized_video[i, ..., 1] = imresize(video[i, ..., 1], size, interpolation)
    elif num_channels == 1:
      resized_video[i, ..., 0] = imresize(video[i, ..., 0], size, interpolation)
    else:
      raise NotImplementedError

  return resized_video.reshape((*shape, *size, video.shape[-1]))


def proc_oflow(images):
  h, w = images.shape[-3:-1]

  processed_images = []
  for image in images:
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255

    mag, ang = cv2.cartToPolar(image[..., 0], image[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    processed_images.append(processed_image)

  return np.stack(processed_images)
