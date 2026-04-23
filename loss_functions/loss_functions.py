# loss_functions/loss_functions.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict
from logger import logger


class RankingMSELoss(nn.Module):
    """
    Combine Mean Squared Error with a margin-ranking loss for pairwise ordering.

    This loss encourages both accurate regression and correct relative ordering
    of affinity predictions across a batch.
    """

    def __init__(self, alpha: float = 2.0) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        logger.info(f"[RankingMSELoss] Initialized with alpha={alpha}")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the combined regression and ranking loss.

        Args:
            pred: Predicted values tensor.
            target: Target values tensor.

        Returns:
            Combined loss tensor.
        """
        mse_loss = self.mse(pred, target)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        diff_true = target_flat.unsqueeze(0) - target_flat.unsqueeze(1)
        diff_pred = pred_flat.unsqueeze(0) - pred_flat.unsqueeze(1)

        mask = diff_true > 0
        ranking_loss = torch.relu(-diff_pred[mask]).mean()
        return mse_loss + self.alpha * ranking_loss


CUSTOM_LOSSES: Dict[str, Any] = {
    "RankingMSELoss": RankingMSELoss,
}


def get_loss_function(config_training: Dict[str, Any]) -> nn.Module:
    """
    Return the requested loss function instance based on configuration.

    Args:
        config_training: Training configuration dictionary with 'loss_fn' key.

    Returns:
        Instantiated loss module.
    """
    loss_name = config_training['loss_fn']

    if loss_name in CUSTOM_LOSSES:
        loss_class = CUSTOM_LOSSES[loss_name]
        if loss_name == "RankingMSELoss":
            return loss_class(alpha=config_training.get('ranking_alpha', 2.0))
        return loss_class()
    if hasattr(nn, loss_name):
        logger.info(f"[loss_functions] Using built-in PyTorch loss {loss_name}")
        return getattr(nn, loss_name)()
    raise ValueError(f"Loss function '{loss_name}' not found in PyTorch or custom registry.")
