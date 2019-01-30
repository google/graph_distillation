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

"""Process Skeleton Feature."""

import numpy as np
import utils


def read_skel(dset, path):
  """
  :param dset: name of dataset, either 'ntu-rgbd' or 'pku-mmd'
  :param path: path to the skeleton file
  :return:
  """
  if dset == 'ntu-rgbd':
    file = open(path, 'r')
    lines = file.readlines()
    num_lines = len(lines)
    num_frames = int(lines[0])
    # print(num_lines, num_frames)

    line_id = 1
    data = []
    for i in range(num_frames):
      num_skels = int(lines[line_id])
      # print(num_skels)

      joints = []
      for _ in range(num_skels):
        num_joints = int(lines[line_id+2])
        # print(num_joints)

        joint = []
        for k in range(num_joints):
          tmp = lines[line_id+3+k].rstrip().split(' ')
          x_3d, y_3d, z_3d, x_depth, y_depth, x_rgb, y_rgb, orientation_w,\
          orientation_x, orientation_y, orientation_z = list(
              map(float, tmp[:-1]))
          joint.append([x_3d, y_3d, z_3d])
        joints.append(joint)
        line_id += 2+num_joints
      joints = np.array(joints)
      data.append(joints)
      line_id += 1

    assert line_id == num_lines

  elif dset == 'pku-mmd':
    file = open(path, 'r')
    lines = file.readlines()
    # num_lines = len(lines)

    data = []
    for line in lines:
      joints = list(map(float, line.rstrip().split(' ')))
      joints = np.array(joints).reshape(2, -1, 3)

      if not np.any(joints[1]):
        joints = joints[0][np.newaxis, :, :]

      data.append(joints)

  elif dset == 'cad-60':
    f = open(path, 'r')
    lines = f.readlines()
    data = []

    # Last line is "END"
    for line in lines[:-1]:
      # fist item is frame number, last item is empty
      row = line.split(',')[1:-1]
      row = list(map(float, row))
      joints = []
      for i in range(15):
        if i < 11:
          # First 11 joints
          index = 14 * i + 10
        else:
          # Joint 12 ~ 15
          index = 11 * 14 + (i - 11) * 4
        joint = row[index: index+3]
        joints.append(joint)
      joints = np.array(joints) / 1000.0  # millimeter to meter
      joints = joints[np.newaxis, :, :]  # To match ntu-rgb format
      data.append(joints)

  else:
    raise NotImplementedError

  return data


def flip_skel(skel, dset):
  """processed skel (normalized and center shifted to the origin)."""
  # Shape: (N x NUM_JOINTS x 3)
  if dset == 'cad-60':
    num_joints = 15
    assert skel.ndim == 3 and skel.shape[1] == num_joints
    assert np.sum(np.mean(skel, axis=(0, 1))) < 1e-8, 'Skeleton not centered.'
    new_skel = skel.copy()
    # Head, neck, torso
    new_skel[:, 0, 0] = -skel[:, 0, 0]
    new_skel[:, 1, 0] = -skel[:, 1, 0]
    new_skel[:, 2, 0] = -skel[:, 2, 0]
    # Shoulder
    new_skel[:, 3, 0] = -skel[:, 5, 0]
    new_skel[:, 5, 0] = -skel[:, 3, 0]
    new_skel[:, 3, 1:] = skel[:, 5, 1:]
    new_skel[:, 5, 1:] = skel[:, 3, 1:]
    # elbow
    new_skel[:, 4, 0] = -skel[:, 6, 0]
    new_skel[:, 6, 0] = -skel[:, 4, 0]
    new_skel[:, 4, 1:] = skel[:, 6, 1:]
    new_skel[:, 6, 1:] = skel[:, 4, 1:]
    # hip
    new_skel[:, 7, 0] = -skel[:, 9, 0]
    new_skel[:, 9, 0] = -skel[:, 7, 0]
    new_skel[:, 7, 1:] = skel[:, 9, 1:]
    new_skel[:, 9, 1:] = skel[:, 7, 1:]
    # knee
    new_skel[:, 8, 0] = -skel[:, 10, 0]
    new_skel[:, 10, 0] = -skel[:, 8, 0]
    new_skel[:, 8, 1:] = skel[:, 10, 1:]
    new_skel[:, 10, 1:] = skel[:, 8, 1:]
    # hand
    new_skel[:, 11, 0] = -skel[:, 12, 0]
    new_skel[:, 12, 0] = -skel[:, 11, 0]
    new_skel[:, 11, 1:] = skel[:, 12, 1:]
    new_skel[:, 12, 1:] = skel[:, 11, 1:]
    # foot
    new_skel[:, 13, 0] = -skel[:, 14, 0]
    new_skel[:, 14, 0] = -skel[:, 13, 0]
    new_skel[:, 13, 1:] = skel[:, 14, 1:]
    new_skel[:, 14, 1:] = skel[:, 13, 1:]
    return new_skel


def pad_skel(skel, axis=1):
  if skel.shape[axis] == 1:
    skel = np.repeat(skel, 2, axis=axis)
  return skel


def extract(skel):
  """Extract. timestep x 2 x num_joints (25) x 3"""
  timestep = skel.shape[0]
  keep_joints = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21]
  skel = skel[:, :, [keep_joint-1 for keep_joint in keep_joints]]
  skel = skel.reshape(timestep, -1, 3)  # timestep x 22 x 3
  num_joints = len(keep_joints)

  jjd, jjv = [], []
  for t in range(timestep):
    jjd_t, jjv_t = [], []
    for i in range(num_joints):
      for j in range(i, num_joints, 1):
        # joint-joint distance
        jjd_t.append(utils.l2_norm(skel[t, i], skel[t, j]))

        # joint-joint vector
        jjv_t.append(skel[t, i]-skel[t, j])
    jjd.append(jjd_t)
    jjv.append(jjv_t)
