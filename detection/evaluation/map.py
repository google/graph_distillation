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

"""Calculate mAP @ IoU thresholds for detection."""

import numpy as np
from ..third_party.pku-mmd.evaluate import process  # TODO(alan) please see whether this work


def get_segments(scores, activity_threshold):
  """Get prediction segments of a video."""
  # Each segment contains start, end, class, confidence score.
  # Sum of all probabilities (1 - probability of no-activity)
  activity_prob = 1 - scores[:, 0]
  # Binary vector indicating whether a clip is an activity or no-activity
  activity_tag = np.zeros(activity_prob.shape, dtype=np.int32)
  activity_tag[activity_prob >= activity_threshold] = 1
  assert activity_tag.ndim == 1
  # For each index, subtract the previous index, getting -1, 0, or 1
  # 1 indicates the start of a segment, and -1 indicates the end.
  padded = np.pad(activity_tag, pad_width=1, mode='constant')
  diff = padded[1:] - padded[:-1]
  indexes = np.arange(diff.size)
  startings = indexes[diff == 1]
  endings = indexes[diff == -1]
  assert startings.size == endings.size

  segments = []
  for start, end in zip(startings, endings):
    segment_scores = scores[start:end, :]
    class_prob = np.mean(segment_scores, axis=0)
    segment_class_index = np.argmax(class_prob[1:]) + 1
    confidence = np.mean(segment_scores[:, segment_class_index])
    segments.append((start, end, segment_class_index, confidence))
  return segments


def calc_map(opt, video_scores, video_names, groundtruth_dir, iou_thresholds):
  """Get mAP (action) for IoU 0.1, 0.3 and 0.5."""
  activity_threshold = 0.4
  num_videos = len(video_scores)
  video_files = [name + '.txt' for name in video_names]

  v_props = []
  for i in range(num_videos):
    # video_name = video_names[i]
    scores = video_scores[i]
    segments = get_segments(scores, activity_threshold)

    prop = []
    for segment in segments:
      start, end, cls, score = segment
      # start, end are indices of clips. Transform to frame index.
      start_index = start * opt.step_size * opt.downsample
      end_index = (
          (end - 1) * opt.step_size + opt.n_frames) * opt.downsample - 1
      prop.append([cls, start_index, end_index, score, video_files[i]])
    v_props.append(prop)

  # Run evaluation on different IoU thresholds.
  mean_aps = []
  for iou in iou_thresholds:
    mean_ap = process(v_props, video_files, groundtruth_dir, iou)
    mean_aps.append(mean_ap)
  return mean_aps
