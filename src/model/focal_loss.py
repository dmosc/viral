import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.95, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Focal Loss for imbalanced binary classification.
        alpha: Weighting factor for the positive class (e.g., 0.95 for 5% positive examples).
        gamma: Focusing parameter to down-weight easy examples (typically 2.0).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Compute standard BCE loss with logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        # 2. Get probabilities from logits
        probs = torch.sigmoid(logits)
        # 3. Calculate p_t (probability of the true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # 4. Calculate alpha_t (weighting for the true class)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # 5. Compute the focal weight and apply it
        focal_weight = alpha_t * torch.pow(1 - p_t, self.gamma)
        focal_loss = focal_weight * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
