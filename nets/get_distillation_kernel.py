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

"""Graph distillation kernel."""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import utils


class DistillationKernel(nn.Module):
  """Graph Distillation kernel.

  Calculate the edge weights e_{j->k} for each j. Modality k is specified by
  to_idx, and the other modalities are specified by from_idx.
  """

  def __init__(self, n_classes, hidden_size, gd_size, to_idx, from_idx,
               gd_prior, gd_reg, w_losses, metric, alpha):
    super(DistillationKernel, self).__init__()
    self.W_logit = nn.Linear(n_classes, gd_size)
    self.W_repr = nn.Linear(hidden_size, gd_size)
    self.W_edge = nn.Linear(gd_size * 4, 1)

    self.gd_size = gd_size
    self.to_idx = to_idx
    self.from_idx = from_idx
    self.alpha = alpha
    # For calculating distillation loss
    self.gd_prior = Variable(torch.FloatTensor(gd_prior).cuda())
    self.gd_reg = gd_reg
    self.w_losses = w_losses  # [logit weight, repr weight]
    self.metric = metric


  def forward(self, logits, reprs):
    """
    Args:
      logits: (n_modalities, batch_size, n_classes)
      reprs: (n_modalities, batch_siz`, hidden_size)
    Return:
      edges: weights e_{j->k} (n_modalities_from, batch_size)
    """
    n_modalities, batch_size = logits.size()[:2]
    z_logits = self.W_logit(logits.view(n_modalities * batch_size, -1))
    z_reprs = self.W_repr(reprs.view(n_modalities * batch_size, -1))
    z = torch.cat(
        (z_logits, z_reprs), dim=1).view(n_modalities, batch_size,
                                         self.gd_size * 2)

    edges = []
    for i in self.from_idx:
      # To calculate e_{j->k}, concatenate z^j, z^k
      e = self.W_edge(torch.cat((z[self.to_idx], z[i]), dim=1))
      edges.append(e)
    edges = torch.cat(edges, dim=1)
    edges = F.softmax(edges * self.alpha, dim=1).transpose(0, 1)
    return edges


  def distillation_loss(self, logits, reprs, edges):
    """Calculate graph distillation losses, which include:
    regularization loss, loss for logits, and loss for representation.
    """
    # Regularization for graph distillation (average across batch)
    loss_reg = (edges.mean(1) - self.gd_prior).pow(2).sum() * self.gd_reg

    loss_logit, loss_repr = 0, 0
    for i, idx in enumerate(self.from_idx):
      w_distill = edges[i] + self.gd_prior[i]  # add graph prior
      loss_logit += self.w_losses[0] * utils.distance_metric(
          logits[self.to_idx], logits[idx], self.metric, w_distill)
      loss_repr += self.w_losses[1] * utils.distance_metric(
          reprs[self.to_idx], reprs[idx], self.metric, w_distill)
    return loss_reg, loss_logit, loss_repr


def get_distillation_kernel(n_classes,
                            hidden_size,
                            gd_size,
                            to_idx,
                            from_idx,
                            gd_prior,
                            gd_reg,
                            w_losses,
                            metric,
                            alpha=1 / 8):
  return DistillationKernel(n_classes, hidden_size, gd_size, to_idx, from_idx,
                            gd_prior, gd_reg, w_losses, metric, alpha)
