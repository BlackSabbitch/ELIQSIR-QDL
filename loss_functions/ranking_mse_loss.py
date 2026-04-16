# loss_functions/ranking_mse_loss.py

import torch
import torch.nn as nn


class RankingMSELoss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        
        # Дифференцируемая аппроксимация CI (Margin Ranking)
        # Берем пары внутри батча
        pred = pred.view(-1)
        target = target.view(-1)
        
        diff_true = target.unsqueeze(0) - target.unsqueeze(1)
        diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)
        
        # Штрафуем, если порядок в предсказаниях не совпадает с таргетом
        # (инверсии в ранжировании)
        mask = diff_true > 0
        ranking_loss = torch.relu(-diff_pred[mask]).mean()
        
        return mse_loss + self.alpha * ranking_loss
