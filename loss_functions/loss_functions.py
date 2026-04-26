# loss_functions/loss_functions.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict
from logger import *

LOSS_ALIASES: Dict[str, str] = {
    "MAELoss": "L1Loss"
}


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
        log_info(f"Initialized with alpha={alpha}", stage="RankingMSELoss")

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


class QuantileLoss(nn.Module):
    """
    Quantile regression loss.

    Penalizes over- and under-estimation asymmetrically, which is useful when
    modeling conservative predictions or skewed distributions.
    """

    def __init__(self, quantile: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < quantile < 1.0:
            raise ValueError("Quantile must be between 0 and 1")
        self.quantile = quantile
        log_info(f"Initialized with quantile={quantile}", stage="QuantileLoss")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = target - pred
        loss = torch.max((self.quantile - 1.0) * diff, self.quantile * diff)
        return loss.mean()


class SlopeRegularizationLoss(nn.Module):
    """
    Wrap a base regression loss with a penalty on the slope of predictions.

    This encourages the model to produce predictions whose linear fit against
    the true labels has a slope closer to the target value, reducing conservative bias.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weight: float = 0.05,
        target_slope: float = 1.0,
        one_sided: bool = True,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weight = weight
        self.target_slope = target_slope
        self.one_sided = one_sided
        log_debug(
            f"Initialized weight={weight}, target_slope={target_slope}, one_sided={one_sided}",
            stage="SlopeRegularizationLoss"
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        base = self.base_loss(pred, target)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
        denominator = ((pred_flat - pred_mean) ** 2).sum() + 1e-8
        denominator = denominator.clamp(min=1e-4)
        slope = numerator / denominator

        if self.one_sided:
            deviation = torch.relu(self.target_slope - slope)
        else:
            deviation = slope - self.target_slope

        slope_penalty = self.weight * deviation * deviation
        return base + slope_penalty


CUSTOM_LOSSES: Dict[str, Any] = {
    "RankingMSELoss": RankingMSELoss,
    "QuantileLoss": QuantileLoss,
}


def get_loss_function(config_training: Dict[str, Any]) -> nn.Module:
    """
    Return the requested loss function instance based on configuration.

    Args:
        config_training: Training configuration dictionary with 'loss_fn' key.

    Returns:
        Instantiated loss module.
    """
    loss_name = config_training['loss_fn']['selected']
    loss_parameters = config_training['loss_fn']['available'].get(loss_name, {})

    if loss_name in LOSS_ALIASES:
        loss_name = LOSS_ALIASES[loss_name]

    if loss_name in CUSTOM_LOSSES:
        loss_class = CUSTOM_LOSSES[loss_name]
        return loss_class(**loss_parameters)

    if hasattr(nn, loss_name):
        log_info(f"Using built-in PyTorch loss {loss_name}", stage="LossFunction")
        loss_module = getattr(nn, loss_name)(**loss_parameters)
    else:
        raise ValueError(f"Loss function '{loss_name}' not found in PyTorch or custom registry.")

    reg_cfg = config_training['loss_fn'].get('regularization', {})
    regularisation_enabled = reg_cfg.get('enabled', False)
    if regularisation_enabled:
        if reg_cfg.get('type', 'slope') == 'slope':
            loss_module = SlopeRegularizationLoss(
                loss_module,
                weight=reg_cfg.get('weight', 0.05),
                target_slope=reg_cfg.get('target_slope', 1.0),
                one_sided=reg_cfg.get('one_sided', True),
            )
        else:
            log_warn(f"Unknown regularization type: {reg_cfg.get('type')} - skipping", stage="LossFunctions")
    log_info(f"Slope regularization: {regularisation_enabled}", stage="LossFunction")

    return loss_module
