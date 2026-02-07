import numpy as np
import torch.linalg as LA
from typing import List, Literal, Dict, Optional
import torch.nn.functional as F
import torch
from torch import nn


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

class TverskyLoss(nn.Module):   
    def __init__(
        self,
        mode: str = "multiclass",
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)
 
        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)
 
        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)
 
        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)

            y_true = F.one_hot(
                (y_true * mask).to(torch.long), num_classes
            )  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
        else:
            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # N, C, H*W


        # 
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

    def _compute_tversky_score(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        dims=None,
    ) -> torch.Tensor:
  
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