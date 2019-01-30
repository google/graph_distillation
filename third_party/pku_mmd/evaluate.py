#    Copyright [2017] [Institute of Computer Science and Technology, Peking University]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Code adopted and modified from
https://github.com/ECHO960/PKU-MMD/blob/master/evaluate.py
"""

import argparse
import os
import numpy as np

number_label = 52


# calc_pr: calculate precision and recall
#    @positive: number of positive proposal
#    @proposal: number of all proposal
#    @ground: number of ground truth
def calc_pr(positive, proposal, ground):
  if (proposal == 0):
    return 0, 0
  if (ground == 0):
    return 0, 0
  return (1.0 * positive) / proposal, (1.0 * positive) / ground


# match: match proposal and ground truth
#    @lst: list of proposals(label, start, end, confidence, video_name)
#    @ratio: overlap ratio
#    @ground: list of ground truth(label, start, end, confidence, video_name)
#
#    correspond_map: record matching ground truth for each proposal
#    count_map: record how many proposals is each ground truth matched by
#    index_map: index_list of each video for ground truth
def match(lst, ratio, ground):

  def overlap(prop, ground):
    l_p, s_p, e_p, c_p, v_p = prop
    l_g, s_g, e_g, c_g, v_g = ground
    if (int(l_p) != int(l_g)):
      return 0
    if (v_p != v_g):
      return 0
    return (min(e_p, e_g) - max(s_p, s_g)) / (max(e_p, e_g) - min(s_p, s_g))

  cos_map = [-1 for x in range(len(lst))]
  count_map = [0 for x in range(len(ground))]
  #generate index_map to speed up
  index_map = [[] for x in range(number_label)]
  for x in range(len(ground)):
    index_map[int(ground[x][0])].append(x)

  for x in range(len(lst)):
    for y in index_map[int(lst[x][0])]:
      if (overlap(lst[x], ground[y]) < ratio):
        continue
      if (overlap(lst[x], ground[y]) < overlap(lst[x], ground[cos_map[x]])):
        continue
      cos_map[x] = y
    if (cos_map[x] != -1):
      count_map[cos_map[x]] += 1
  positive = sum([(x > 0) for x in count_map])
  return cos_map, count_map, positive


# Interpolated Average Precision:
#    @lst: list of proposals(label, start, end, confidence, video_name)
#    @ratio: overlap ratio
#    @ground: list of ground truth(label, start, end, confidence, video_name)
#
#    score = sigma(precision(recall) * delta(recall))
#    Note that when overlap ratio < 0.5,
#        one ground truth will correspond to many proposals
#        In that case, only one positive proposal is counted
def ap(lst, ratio, ground):
  lst.sort(key=lambda x: x[3])  # sorted by confidence
  cos_map, count_map, positive = match(lst, ratio, ground)
  score = 0
  number_proposal = len(lst)
  number_ground = len(ground)
  old_precision, old_recall = calc_pr(positive, number_proposal, number_ground)

  for x in range(len(lst)):
    number_proposal -= 1
    if (cos_map[x] == -1):
      continue
    count_map[cos_map[x]] -= 1
    if (count_map[cos_map[x]] == 0):
      positive -= 1

    precision, recall = calc_pr(positive, number_proposal, number_ground)
    if precision > old_precision:
      old_precision = precision
    score += old_precision * (old_recall - recall)
    old_recall = recall
  return score


def process(v_props, video_files, groundtruth_dir, theta):
  v_grounds = []  # ground-truth list separated by video

  #========== find all proposals separated by video========
  for video in video_files:
    ground = open(os.path.join(groundtruth_dir, video), "r").readlines()
    ground = [ground[x].replace(",", " ") for x in range(len(ground))]
    ground = [[float(y) for y in ground[x].split()] for x in range(len(ground))]
    #append video name
    for x in ground:
      x.append(video)
    v_grounds.append(ground)

  assert len(v_props) == len(v_grounds), "{} != {}".format(
      len(v_props), len(v_grounds))

  #========== find all proposals separated by action categories========
  # proposal list separated by class
  a_props = [[] for x in range(number_label)]
  # ground-truth list separated by class
  a_grounds = [[] for x in range(number_label)]

  for x in range(len(v_props)):
    for y in range(len(v_props[x])):
      a_props[int(v_props[x][y][0])].append(v_props[x][y])

  for x in range(len(v_grounds)):
    for y in range(len(v_grounds[x])):
      a_grounds[int(v_grounds[x][y][0])].append(v_grounds[x][y])

  #========== find all proposals========
  all_props = sum(a_props, [])
  all_grounds = sum(a_grounds, [])

  #========== calculate protocols========
  # mAP_action
  aps_action = np.array([
      ap(a_props[x + 1], theta, a_grounds[x + 1])
      for x in range(number_label - 1)
  ])
  # mAP_video
  aps_video = np.array(
      [ap(v_props[x], theta, v_grounds[x]) for x in range(len(v_props))])
  # Return mAP action.
  return np.mean(aps_action)
