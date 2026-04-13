# evaluator.py

from scipy.stats import pearsonr
import torch
import numpy as np
from matplotlib import pyplot as plt


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

        return rmse, r_val, ci_val, preds, targets

    def plot_history(self, exp_dir, show=True, save=True):
        epochs = range(1, len(self.model.history['train_loss']) + 1)
        
        plt.figure(figsize=(15, 15))

        # 1. Loss (Обучение)
        plt.subplot(3, 2, 1)
        plt.plot(epochs, self.model.history['train_loss'], 'b-', label='Train Loss')
        plt.title('Learning Curve (Loss)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)

        # 2. RMSE
        plt.subplot(3, 2, 2)
        plt.plot(epochs, self.model.history['val_rmse'], 'r-', label='Val RMSE')
        plt.title('Validation RMSE')
        plt.xlabel('Epochs')
        plt.grid(True)

        # 3. Pearson R
        plt.subplot(3, 2, 3)
        plt.plot(epochs, self.model.history['val_pearson'], 'g-', label='Pearson R')
        plt.title('Correlation (Pearson R)')
        plt.xlabel('Epochs')
        plt.ylabel('R')
        plt.grid(True)

        # 4. CI
        plt.subplot(3, 2, 4)
        plt.plot(epochs, self.model.history['val_ci'], 'm-', label='Concordance Index')
        plt.title('Ranking Accuracy (CI)')
        plt.xlabel('Epochs')
        plt.ylabel('CI')
        plt.grid(True)

        y_true_np = np.array(self.model.history['best_y_true'])
        y_pred_np = np.array(self.model.history['best_y_pred'])

        plt.subplot(3, 2, 5)
        plt.scatter(y_true_np, y_pred_np, alpha=0.5, color='teal')
        # Линия идеального предсказания (диагональ)
        lims = [min(min(y_true_np), min(y_pred_np)), max(max(y_true_np), max(y_pred_np))]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        plt.title(f'Actual vs Predicted (Epoch {len(self.model.history["val_pearson"])})')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.grid(True)

        # 3. Распределение ошибки (Гистограмма)
        plt.subplot(3, 2, 6)
        errors = y_pred_np - y_true_np
        plt.hist(errors, bins=20, color='salmon', edgecolor='black')
        plt.axvline(0, color='black', linestyle='--')
        plt.title('Error Distribution (Residuals)')
        plt.xlabel('Prediction Error')
        plt.grid(True)

        plt.tight_layout()
        if save:
            plt.savefig(f'{exp_dir}/model_performance_report.png')
        if show:
            plt.show()
