# evaluator.py

from scipy.stats import pearsonr
import torch
import numpy as np


class EValuator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @staticmethod
    def concordance_index(y_true, y_pred):
        """Векторизованная версия CI через NumPy"""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Создаем матрицы разностей: diff[i, j] = y[i] - y[j]
        # Это создает матрицы NxN
        true_diff = y_true[:, np.newaxis] - y_true[np.newaxis, :]
        pred_diff = y_pred[:, np.newaxis] - y_pred[np.newaxis, :]
        
        # Нам нужны только пары, где y_true[i] > y_true[j]
        mask = true_diff > 0
        valid_pairs = mask.sum()
        
        if valid_pairs == 0:
            return 0.5
        
        # Считаем совпадения порядков
        # 1.0 если предсказание тоже >
        # 0.5 если предсказание равно
        concordant = (pred_diff[mask] > 0).sum()
        ties = (pred_diff[mask] == 0).sum()
        
        return (concordant + 0.5 * ties) / valid_pairs

    def evaluate(self, loader):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for prot, lig, pock, y in loader:
                y_hat = self.model(prot.to(self.device), lig.to(self.device), pock.to(self.device))
                preds.extend(y_hat.cpu().view(-1).tolist())
                targets.extend(y.tolist())
        
        preds = np.array(preds)
        targets = np.array(targets)
        
        rmse = np.sqrt(np.mean((preds - targets)**2))
        r_val, _ = pearsonr(preds, targets)
        ci_val = self.concordance_index(targets, preds) # Считаем CI

        return rmse, r_val, ci_val
