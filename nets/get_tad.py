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

"""Model for temporal action detection."""

import torch.nn as nn
from third_party.pytorch.get_cnn import *
from .get_gru import *


class TAD(nn.Module):
  """Temporal action detection model.

  Consists of visual encoder (specified by encoder_type) and sequence
  encoder.
  """

  def __init__(self, n_classes, n_frames, n_channels, input_size, hidden_size,
               n_layers, dropout, hidden_size_seq, n_layers_seq, dropout_seq,
               encoder_type):
    super(TAD, self).__init__()

    # Visual encoder
    if encoder_type == 'cnn':
      self.embed = get_resnet(n_frames * n_channels, n_classes)
    elif encoder_type == 'rnn':
      self.embed = get_gru(input_size, hidden_size, n_layers, dropout,
                           n_classes)
    else:
      raise NotImplementedError

    # Sequence encoder
    self.rnn = nn.GRU(
        input_size=hidden_size,
        hidden_size=hidden_size_seq,
        num_layers=n_layers_seq,
        batch_first=True,
        dropout=dropout_seq,
        bidirectional=True)
    # Classification layer
    self.fc = nn.Linear(hidden_size_seq,
                        n_classes + 1)  # plus 1 class for background

    self.n_classes = n_classes
    self.hidden_size = hidden_size
    self.hidden_size_seq = hidden_size_seq
    self.n_layers_seq = n_layers_seq
    self.encoder_type = encoder_type

  def forward(self, x):
    """:param x: if encoder_type == 'cnn', batch_size x timestep x n_frames x n_channels x h x w

              if encoder_type == 'lstm', batch_size x timestep x n_frames x
              input_size
    :return: representation and logits
    """
    if self.encoder_type == 'cnn':
      batch_size, timestep, n_frames, n_channels, h, w = x.size()
      x = x.view(batch_size * timestep, n_frames * n_channels, h, w)
      _, x = self.embed(x)
      x = x.view(batch_size, timestep, self.hidden_size)
    elif self.encoder_type == 'rnn':
      batch_size, timestep, n_frames, input_size = x.size()
      x = x.view(batch_size * timestep, n_frames, input_size)
      _, x = self.embed(x)
      x = x.view(batch_size, timestep, self.hidden_size)
    else:
      raise NotImplementedError

    batch_size = x.size(0)
    x, _ = self.rnn(x)
    x = x.contiguous().view(x.size(0), x.size(1), 2, -1).sum(2)
    representation = x
    logit = self.fc(representation)

    return logit, representation


def get_tad(n_classes, n_frames, n_channels, input_size, hidden_size, n_layers,
            dropout, hidden_size_seq, n_layers_seq, dropout_seq, encoder_type):
  return TAD(n_classes, n_frames, n_channels, input_size, hidden_size, n_layers,
             dropout, hidden_size_seq, n_layers_seq, dropout_seq, encoder_type)
