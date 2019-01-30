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

"""Get detection model."""

from .model import SingleStream
from .model import GraphDistillation

ALL_MODALITIES = ['rgb', 'oflow', 'depth', 'jjd', 'jjv', 'jld']


def get_model(opt):
  if opt.dset == 'pku-mmd':
    n_classes = 51
    all_input_sizes = [-1, -1, -1, 276, 828, 836]
    all_n_channels = [3, 2, 1, -1, -1, -1]
  else:
    raise NotImplementedError

  n_channels = [all_n_channels[ALL_MODALITIES.index(m)] for m in opt.modalities]
  input_sizes = [
      all_input_sizes[ALL_MODALITIES.index(m)] for m in opt.modalities
  ]

  if len(opt.modalities) == 1:
    # Single stream
    index = 0
    model = SingleStream(opt.modalities, n_classes, opt.n_frames, n_channels,
                         input_sizes, opt.hidden_size, opt.n_layers,
                         opt.dropout, opt.hidden_size_seq, opt.n_layers_seq,
                         opt.dropout_seq, opt.bg_w, opt.lr, opt.lr_decay_rate,
                         index, opt.ckpt_path)
  else:
    index = opt.modalities.index(opt.xfer_to)
    model = GraphDistillation(
        opt.modalities, n_classes, opt.n_frames, n_channels, input_sizes,
        opt.hidden_size, opt.n_layers, opt.dropout, opt.hidden_size_seq,
        opt.n_layers_seq, opt.dropout_seq, opt.bg_w, opt.lr, opt.lr_decay_rate,
        index, opt.ckpt_path, opt.w_losses, opt.w_modalities, opt.metric,
        opt.xfer_to, opt.gd_size, opt.gd_reg)

  return model
