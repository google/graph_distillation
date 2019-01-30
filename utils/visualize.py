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

import os
import cv2
import numpy as np
import utils
from utils import imgproc


def visualize_rgb(images):
  """Visualize RGB modality."""
  images = utils.to_numpy(images)

  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  images = np.moveaxis(images, -3, -1)
  images = images*std+mean
  images = np.clip(images*255, 0, 255)
  images = images[..., ::-1].astype(np.uint8)
  images = images[0, 0]  # subsample

  imgproc.save_avi('/home/luoa/research/rgb.avi', images)


def visualize_oflow(images):
  """Visualize optical flow modality."""
  images = utils.to_numpy(images)

  images = np.moveaxis(images, -3, -1)
  images = images[0, 0]  # subsample

  images = imgproc.proc_oflow(images)
  imgproc.save_avi('/home/luoa/research/oflow.avi', images)


def visualize_warp(rgb, oflow):
  """TODO: add info."""
  rgb = utils.to_numpy(rgb)
  oflow = utils.to_numpy(oflow)

  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  rgb = np.moveaxis(rgb, -3, -1)
  rgb = rgb*std+mean
  rgb = np.clip(rgb*255, 0, 255)
  bgr = rgb[..., ::-1].astype(np.uint8)
  bgr = bgr[0, 0]  # subsample
  print(bgr.shape, np.amin(bgr), np.amax(bgr), np.mean(bgr),
        np.mean(np.absolute(bgr)))

  oflow = np.moveaxis(oflow, -3, -1)
  oflow = oflow[0, 0]  # subsample
  print(oflow.shape, np.amin(oflow), np.amax(oflow), np.mean(oflow),
        np.mean(np.absolute(oflow)))

  warp = imgproc.warp(bgr[4], bgr[5], oflow[4])

  root = '/home/luoa/research'
  cv2.imwrite(os.path.join(root, 'bgr1.jpg'), bgr[4])
  cv2.imwrite(os.path.join(root, 'bgr2.jpg'), bgr[5])
  cv2.imwrite(os.path.join(root, 'warp.jpg'), warp)
