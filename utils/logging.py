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

"""Calculate result statistics, logging."""

from collections import OrderedDict
import logging
import os
import sys
import numpy as np


class _AverageMeter(object):
  """ Average Meter Class."""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count


class Statistics(object):
  """ Statistics Class."""

  def __init__(self, ckpt_path=None, name='history'):
    self.meters = OrderedDict()
    self.history = OrderedDict()
    self.ckpt_path = ckpt_path
    self.name = name

  def update(self, n, ordered_dict):
    info = ''
    for key in ordered_dict:
      if key not in self.meters:
        self.meters.update({key: _AverageMeter()})
      self.meters[key].update(ordered_dict[key], n)
      info += '{key}={var.val:.4f}, avg {key}={var.avg:.4f}, '.format(
          key=key, var=self.meters[key])

    return info[:-2]

  def summarize(self, reset=True):
    info = ''
    for key in self.meters:
      info += '{key}={var:.4f}, '.format(key=key, var=self.meters[key].avg)

    if reset:
      self.reset()

    return info[:-2]

  def reset(self):
    for key in self.meters:
      if key in self.history:
        self.history[key].append(self.meters[key].avg)
      else:
        self.history.update({key: [self.meters[key].avg]})

    self.meters = OrderedDict()

  def load(self):
    self.history = np.load(
        os.path.join(self.ckpt_path, '{}.npy'.format(self.name))).item()

  def save(self):
    np.save(
        os.path.join(self.ckpt_path, '{}.npy'.format(self.name)), self.history)


class Logger(object):
  """ Logger Class."""

  def __init__(self, path, name='debug'):
    self.logger = logging.getLogger()
    self.logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    fh = logging.FileHandler(os.path.join(path, '{}.log'.format(name)), 'w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    self.logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    self.logger.addHandler(ch)

  def log(self, info):
    self.logger.info(info)
