# loss_functions/loss_functions.py

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


CUSTOM_LOSSES = {
    "RankingMSELoss": RankingMSELoss,
}


def get_loss_function(config_training):
    """Фабрика для выдачи правильной функции потерь"""
    loss_name = config_training['loss_fn']
    
    # 1. Ищем в нашем кастомном реестре
    if loss_name in CUSTOM_LOSSES:
        loss_class = CUSTOM_LOSSES[loss_name]
        
        # Если лоссу нужны спец. параметры, передаем их
        if loss_name == "RankingMSELoss":
            return loss_class(alpha=config_training.get('ranking_alpha', 2.0))
        
        return loss_class() # Вызов по умолчанию
        
    # 2. Ищем в стандартной библиотеке PyTorch
    elif hasattr(nn, loss_name):
        return getattr(nn, loss_name)()
        
    else:
        raise ValueError(f"Loss-функция '{loss_name}' не найдена ни в PyTorch, ни в кастомном реестре.")
