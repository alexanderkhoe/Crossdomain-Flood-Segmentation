import numpy as np
import torch.linalg as LA
from typing import List, Literal, Dict, Optional
import torch.nn.functional as F
import torch
from torch import nn

torch.manual_seed(124)

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

class SpatialTversky(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-7,gamma=0.5, smooth=1e-6, ignore_index=255):
        super(SpatialTversky, self).__init__()
        self.alpha = alpha
        self.beta_base = beta
        self.ignore_index = ignore_index
        self.eps = eps
        self.gamma = gamma
        self.smooth = smooth


    def _compute_tversky_score(self, output, target, dims):
  
        assert output.size() == target.size()

        if dims is not None:
            output_sum = torch.sum(output, dim=dims)
            target_sum = torch.sum(target, dim=dims)
            difference = LA.vector_norm(output - target, ord=1, dim=dims)
        else:
            output_sum = torch.sum(output)
            target_sum = torch.sum(target)
            difference = LA.vector_norm(output - target, ord=1)

 
        intersection = (output_sum + target_sum - difference) / 2   
        fp = output_sum - intersection
        fn = target_sum - intersection
 
        tversky_score = (intersection + self.smooth) / (
            intersection + self.alpha * fp + self.beta * fn + self.smooth
        ).clamp_min(self.eps)
        
        return tversky_score

    def forward(self, y_pred, y_true, water_occurrence):

        assert y_true.size(0) == y_pred.size(0)

        y_pred = y_pred.log_softmax(dim=1).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)

        y_pred = y_pred.view(bs,-1)
        y_true = y_true.view(bs, num_classes, -1)
        dims = (0, 2)

        # mask idx
        mask = y_true != self.ignore_index
        y_pred = y_pred * mask.unsqueeze(1)

        y_true = F.one_hot(
            (y_true * mask).to(torch.long), num_classes
        )  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W


        #

        # occ = water_occurrence.view(-1)
 
        # tp = (y_pred * y_true).sum()
 
        # fp_pixels = y_pred * (1 - y_true)
        # fn_pixels = (1 - y_pred) * y_true
 
        # beta_dynamic = self.beta_base + self.gamma * (1 - occ)
 
        # total_fp = (self.alpha * fp_pixels).sum()
        # total_fn = (beta_dynamic * fn_pixels).sum()
 
        # tversky_index = (tp + self.smooth) / (tp + total_fp + total_fn + self.smooth)

        # return 1 - tversky_index


        scores = self._compute_tversky_score(
            y_pred, y_true.type_as(y_pred), dims=dims
        )
 
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
 
        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)
 
        if self.classes is not None:
            loss = loss[self.classes]
 
        return (loss.mean() ** self.gamma)

 