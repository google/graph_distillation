# MIT License
#
# Copyright (c) 2017 Yi Zhu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Transformations for videos.

Code adapted from
https://github.com/bryanyzhu/two-stream-pytorch/blob/master/video_transforms.py
"""

import collections
import math
import numbers
import random

import numpy as np
import torch
import torch.nn.functional as F

import utils
from utils import imgproc


class ToTensor(object):
  """Converts a numpy.ndarray (...

  x H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (... x C x H x W) in the range [0.0,
  1.0].
  """

  def __init__(self, scale=True, to_float=True):
    self.scale = scale
    self.to_float = to_float

  def __call__(self, arr):
    if isinstance(arr, np.ndarray):
      video = torch.from_numpy(np.rollaxis(arr, axis=-1, start=-3))

      if self.to_float:
        video = video.float()

      if self.scale:
        return video.div(255)
      else:
        return video
    else:
      raise NotImplementedError


class Normalize(object):
  """Given mean and std,
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std
  """

  def __init__(self, mean, std):
    if not isinstance(mean, list):
      mean = [mean]
    if not isinstance(std, list):
      std = [std]

    self.mean = torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)
    self.std = torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)

  def __call__(self, tensor):
    return tensor.sub_(self.mean).div_(self.std)


class Scale(object):
  """Rescale the input numpy.ndarray to the given size.
  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size * height / width, size)
      interpolation (int, optional): Desired interpolation. Default is
          ``bilinear``
  """
  def __init__(self, size, transform_pixel=False, interpolation='bilinear'):
    """:param size: output size :param transform_pixel: transform pixel values for flow :param interpolation: 'bilinear', 'nearest'
    """
    assert isinstance(size, int) or (isinstance(size, collections.Iterable) and
                                     len(size) == 2)
    self.size = size
    self.transform_pixel = transform_pixel
    self.interpolation = interpolation

  def __call__(self, video):
    """Args: video (numpy.ndarray): Video to be scaled.

    Returns:
        numpy.ndarray: Rescaled video.
    """
    w, h = video.shape[-2], video.shape[-3]

    if isinstance(self.size, int):
      if (w <= h and w == self.size) or (h <= w and h == self.size):
        return video

      if w < h:
        ow = self.size
        oh = int(self.size * h / w)
        video = imgproc.resize(video, (oh, ow), self.interpolation)
      else:
        oh = self.size
        ow = int(self.size * w / h)
        video = imgproc.resize(video, (oh, ow), self.interpolation)

      if self.transform_pixel:
        video[..., 0] = (video[..., 0] - 128) * (ow / w) + 128
        video[..., 1] = (video[..., 1] - 128) * (oh / h) + 128
    else:
      video = imgproc.resize(video, self.size, self.interpolation)

      if self.transform_pixel:
        video[..., 0] = (video[..., 0] - 128) * (self.size / w) + 128
        video[..., 1] = (video[..., 1] - 128) * (self.size / h) + 128

    return video


class CenterCrop(object):
  """Crops the given numpy.ndarray at the center to have a region of
  the given size. size can be a tuple (target_height, target_width)
  or an integer, in which case the target will be of a square shape (size, size)
  """

  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, video):
    h, w = video.shape[-3:-1]
    th, tw = self.size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    return video[..., y1:y1 + th, x1:x1 + tw, :]


class Pad(object):
  """Pad the given np.ndarray on all sides with the given "pad" value.

  Args: padding (int or sequence): Padding on each border. If a sequence of
  length 4, it is used to pad left, top, right and bottom borders respectively.
      fill: Pixel fill value. Default is 0.
  """

  def __init__(self, padding, fill=0):
    assert isinstance(padding, numbers.Number)
    assert isinstance(fill, numbers.Number) or isinstance(
        fill, str) or isinstance(fill, tuple)
    self.padding = padding
    self.fill = fill

  def __call__(self, video):
    """Args: video (np.ndarray): Video to be padded.

    Returns:
        np.ndarray: Padded video.
    """
    pad_width = ((0, 0), (self.padding, self.padding), (self.padding,
                                                        self.padding), (0, 0))
    return np.pad(
        video, pad_width=pad_width, mode='constant', constant_values=self.fill)


class RandomCrop(object):
  """Crop the given numpy.ndarray at a random location.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
      padding (int or sequence, optional): Optional padding on each border
          of the image. Default is 0, i.e no padding. If a sequence of length
          4 is provided, it is used to pad left, top, right, bottom borders
          respectively.
  """

  def __init__(self, size, padding=0):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size
    self.padding = padding

  def __call__(self, video):
    """Args: video (np.ndarray): Video to be cropped.

    Returns:
        np.ndarray: Cropped video.
    """
    if self.padding > 0:
      pad = Pad(self.padding, 0)
      video = pad(video)

    w, h = video.shape[-2], video.shape[-3]
    th, tw = self.size
    if w == tw and h == th:
      return video

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return video[..., y1:y1 + th, x1:x1 + tw, :]


class RandomHorizontalFlip(object):
  """Randomly horizontally flips the given numpy.ndarray with a probability of 0.5

  """

  def __init__(self, transform_pixel=False):
    """:param transform_pixel: transform pixel values for flow
    """
    self.transform_pixel = transform_pixel if isinstance(
        transform_pixel, list) else [transform_pixel]

  def __call__(self, videos):
    """Support joint transform
    :param videos: np.ndarray or a list of np.ndarray
    :return:
    """
    if random.random() < 0.5:
      videos = utils.unsqueeze(videos)
      ret = []
      for tp, video in zip(self.transform_pixel, videos):
        video = video[..., ::-1, :]
        if tp:
          video[..., 0] = 255 - video[..., 0]
        ret.append(video.copy())
      return utils.squeeze(ret)
    else:
      return videos


class RandomSizedCrop(object):
  """Crop the given np.ndarray to random size and aspect ratio.
  A crop of random size of (0.4 to 1.0) of the original size and a random
  aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
  is finally resized to given size.
  This is popularly used to train the Inception networks.
  """

  def __init__(self, size, transform_pixel=False):
    """:param size: size of the smaller edge :param transform_pixel: transform pixel values for flow
    """
    self.size = size
    self.transform_pixel = transform_pixel if isinstance(
        transform_pixel, list) else [transform_pixel]

  def __call__(self, videos):
    """Support joint transform
    :param videos: np.ndarray or a list of np.ndarray
    :return:
    """
    videos = utils.unsqueeze(videos)
    h_orig, w_orig = videos[0].shape[-3:-1]

    for attempt in range(10):
      ret = []

      area = h_orig * w_orig
      target_area = random.uniform(0.4, 1.0) * area
      aspect_ratio = random.uniform(3. / 4, 4. / 3)

      w = int(round(math.sqrt(target_area * aspect_ratio)))
      h = int(round(math.sqrt(target_area / aspect_ratio)))

      if random.random() < 0.5:
        w, h = h, w

      if w <= w_orig and h <= h_orig:
        x1 = random.randint(0, w_orig - w)
        y1 = random.randint(0, h_orig - h)

        for tp, video in zip(self.transform_pixel, videos):
          video = video[..., y1:y1 + h, x1:x1 + w, :]
          video = imgproc.resize(video, (self.size, self.size), 'bilinear')
          if tp:
            video[..., 0] = (video[..., 0] - 128) * (self.size / w) + 128
            video[..., 1] = (video[..., 1] - 128) * (self.size / h) + 128

          ret.append(video)

        return utils.squeeze(ret)

    # Fallback
    ret = []
    scales = [Scale(self.size, tp, 'bilinear') for tp in self.transform_pixel]
    crop = CenterCrop(self.size)
    for scale, video in zip(scales, videos):
      video = crop(scale(video))
      ret.append(video)

    return utils.squeeze(ret)


class SkelNormalize(object):
  """Given mean and std, will normalize the numpy array
  """

  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, skel):
    return (skel - self.mean) / self.std


class AvgPool(object):
  """Rescale the input by performing average pooling The height and width are scaled down by a factor of kernel_size
  """

  def __init__(self, kernel_size):
    self.kernel_size = kernel_size

  def __call__(self, x):
    batch_size, n_samples, n_channels, h, w = x.size()
    x = x.view(-1, n_channels, h, w)
    x = F.avg_pool2d(x, self.kernel_size, stride=self.kernel_size).data
    return x.view(batch_size, n_samples, *x.size()[-3:])


class Clip(object):
  """Clip values of the numpy array
  """

  def __init__(self, lower, upper):
    self.lower = lower
    self.upper = upper

  def __call__(self, x):
    return np.clip(x, self.lower, self.upper)
