import torch
import torch.nn as nn
from torch import Tensor


class SoftDiceLoss(nn.Module):
    def __init__(self, aggregate="mean", smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        assert aggregate in ["mean", "min", "max"] or aggregate is None
        self.aggregate = aggregate
        self.smooth = smooth

    def forward(self, probs: Tensor, targets: Tensor):
        """
        probs: Tensor of shape (B, N, H, W, D) containing N predicted probability maps
        target: Tensor of shape (B, H, W, D) containing the ground truth map
        """
        probs = probs.float()
        targets = targets.float()

        if len(probs.shape) == 4:
            probs = probs.permute(1, 0, 2, 3)
            targets = targets.unsqueeze(0).repeat(probs.shape[0], 1, 1, 1)
        elif len(probs.shape) == 5:
            probs = probs.permute(1, 0, 2, 3, 4)
            targets = targets.unsqueeze(0).repeat(probs.shape[0], 1, 1, 1, 1)

        spatial_dims = tuple(range(2, probs.ndim))

        numerator = 2 * torch.sum(probs * targets, dim=spatial_dims)
        denominator = torch.sum(probs ** 2, dim=spatial_dims) + torch.sum(targets ** 2, dim=spatial_dims)

        dice = (numerator + self.smooth) / (denominator + self.smooth)

        if self.aggregate is None:
            return 1 - dice

        if self.aggregate == "max":
            dice_best, idx_best = torch.max(dice, dim=0)
        elif self.aggregate == "min":
            dice_best, idx_best = torch.min(dice, dim=0)
        else:
            dice_best = torch.mean(dice, dim=0)

        return 1 - dice_best.mean()


class DiceLoss(nn.Module):
    def __init__(self, aggregate="mean", threshold=0.5, smooth=1e-6):
        super(DiceLoss, self).__init__()
        assert aggregate in ["mean", "min", "max"] or aggregate is None
        self.aggregate = aggregate
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, probs: Tensor, targets: Tensor):
        """
        probs: Tensor of shape (B, N, H, W, D) containing N predicted probability maps
        target: Tensor of shape (B, H, W, D) containing the ground truth map
        """
        probs = probs.float()
        targets = targets.float()

        if len(probs.shape) == 4:
            probs = probs.permute(1, 0, 2, 3)
            targets = targets.unsqueeze(0).repeat(probs.shape[0], 1, 1, 1)
        elif len(probs.shape) == 5:
            probs = probs.permute(1, 0, 2, 3, 4)
            targets = targets.unsqueeze(0).repeat(probs.shape[0], 1, 1, 1, 1)

        spatial_dims = tuple(range(2, probs.ndim))

        probs = (probs > self.threshold).float()

        numerator = 2 * torch.sum(probs * targets, dim=spatial_dims)
        denominator = torch.sum(probs, dim=spatial_dims) + torch.sum(targets, dim=spatial_dims)

        dice = (numerator + self.smooth) / (denominator + self.smooth)

        if self.aggregate is None:
            return 1 - dice

        if self.aggregate == "max":
            dice_best, idx_best = torch.max(dice, dim=0)
        elif self.aggregate == "min":
            dice_best, idx_best = torch.min(dice, dim=0)
        else:
            dice_best = torch.mean(dice, dim=0)

        return 1 - dice_best.mean()
