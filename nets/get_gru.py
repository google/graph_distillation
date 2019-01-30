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

"""GRU module."""

import torch
from torch.autograd import Variable
import torch.nn as nn


class GRU(nn.Module):
  """ GRU Class"""

  def __init__(self, input_size, hidden_size, n_layers, dropout, n_classes):
    super(GRU, self).__init__()
    self.gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=n_layers,
        batch_first=True,
        dropout=dropout)
    self.fc = nn.Linear(hidden_size, n_classes)

    self.hidden_size = hidden_size
    self.n_layers = n_layers

  def _get_states(self, batch_size):
    h0 = Variable(
        torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda(),
        requires_grad=False)
    return h0

  def forward(self, x):
    """:param x: input of size batch_size' x n_frames x input_size (batch_size' = batch_size*n_samples) :return:
    """
    batch_size = x.size(0)
    h0 = self._get_states(batch_size)
    x, _ = self.gru(x, h0)
    representation = x.mean(1)
    logit = self.fc(representation)
    return logit, representation


def get_gru(input_size, hidden_size, n_layers, dropout, n_classes):
  return GRU(input_size, hidden_size, n_layers, dropout, n_classes)
