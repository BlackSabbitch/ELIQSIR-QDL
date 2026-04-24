import torch
import torch.nn as nn
from tqdm import tqdm
from model.model import UniversalHybridSlotModel
from encoders.original_quantum_encoder import QuantumReUploadingLayer
from utils import Utils
from loss_functions.loss_functions import get_loss_function
from logger import log_info, log_warn
import json
import os
from typing import Tuple
from evaluator import Evaluator


class HybridTrainer:
    """
    Trainer class for hybrid classical-quantum models.

    Handles training loop with separate optimizers for classical and quantum components,
    validation, and experiment tracking. Supports different loss functions and
    learning rate schedules.

    Attributes:
        model: The UniversalHybridSlotModel to train.
        device: Device for training (cuda/cpu).
        config: Training configuration dictionary.
        classic_optimizer: Optimizer for classical parameters.
        quantum_optimizer: Optimizer for quantum parameters (if quantum branch enabled).

    Example:
        >>> trainer = HybridTrainer(model, config, device='cuda')
        >>> trainer.train(train_loader, val_loader)
    """

    def __init__(self, model: UniversalHybridSlotModel, evaluator: Evaluator, config: dict, device: str = 'cuda') -> None:
        """
        Initialize the hybrid trainer.

        Args:
            model: Model instance to train.
            config: Configuration dictionary with training parameters.
            device: Device for training ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.train_cfg = config['training']
        self.evaluator = evaluator or Evaluator(model, device)

        # 1. Separate parameters for optimizers

        classic_params = []
        quantum_params = []

        # Рекурсивно обходим модель
        for module in model.modules():
            # Если это квантовое ядро (неважно где оно стоит - в голове или в ветке)
            if isinstance(module, QuantumReUploadingLayer):
                quantum_params.extend(list(module.parameters()))
            # Линейные слои, GNN и прочее - в классику
            elif len(list(module.children())) == 0: # Только листовые модули (Linear, Conv и т.д.)
                if not any(p is params for params in quantum_params for p in module.parameters()):
                    classic_params.extend(list(module.parameters()))

        # 2. Initialize optimizers from config
        c_opt_cfg = self.train_cfg['optimizers']['classic']
        self.opt_classic = getattr(torch.optim, c_opt_cfg['type'])(classic_params, **c_opt_cfg['params'])
        log_info(f"Classic: {c_opt_cfg['type']} with {len(classic_params)} parameters", stage="OPTIMIZER")
        
        q_opt_cfg = self.train_cfg['optimizers']['quantum']
        if len(quantum_params) > 0:
            self.opt_quantum = getattr(torch.optim, q_opt_cfg['type'])(quantum_params, **q_opt_cfg['params'])
            log_info(f"Quantum: {q_opt_cfg['type']} with {len(quantum_params)} parameters", stage="OPTIMIZER")
        else:
            self.opt_quantum = None
            log_warn(f"Quantum: No quantum parameters found, optimizer disabled", stage="OPTIMIZER")

        self.sched_classic = self._build_scheduler(self.opt_classic, c_opt_cfg.get('scheduler'))
        self.sched_quantum = self._build_scheduler(self.opt_quantum, q_opt_cfg.get('scheduler')) if self.opt_quantum else None

        # 3. Loss function
        self.criterion = get_loss_function(config['training'])
        log_info(f"Используем Loss: {config['training']['loss_fn']['selected']}", stage="TRAINER")

    def _build_scheduler(self, optimizer, sched_cfg):
        if not sched_cfg:
            return None
        # Например: ReduceLROnPlateau или CosineAnnealingLR
        sched_cls = getattr(torch.optim.lr_scheduler, sched_cfg['type'])
        params = sched_cfg.get('params', {}) or {}
        valid_params = Utils.filter_kwargs(sched_cls.__init__, params)
        invalid_params = [key for key in params if key not in valid_params]
        if invalid_params:
            log_warn(
                f"Ignoring unsupported params for {sched_cfg['type']}: {invalid_params}",
                stage="SCHEDULER"
            )
        return sched_cls(optimizer, **valid_params)

    def step_schedulers(self, metrics):
        """Обновление шага обучения"""
        if self.sched_classic:
            # ReduceLROnPlateau требует метрику (loss)
            if isinstance(self.sched_classic, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.sched_classic.step(metrics)
            else:
                self.sched_classic.step()
        
        if self.sched_quantum and self.opt_quantum:
            self.sched_quantum.step()

    def train_epoch(self, loader) -> float:
        """
        Train for one epoch.

        Args:
            loader: DataLoader for training data.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc="Training", unit="batch", leave=True)
        
        for i, batch in enumerate(pbar):
            # Transfer data to device (remember tuple structure)
            batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]
            _, _, _, _, targets = batch
            
            self.opt_classic.zero_grad()
            if self.opt_quantum:
                self.opt_quantum.zero_grad()
            
            # Forward pass
            preds = self.model(batch).squeeze()
            loss = self.criterion(preds, targets)
            
            # Backward pass
            loss.backward()
            
            # Step both optimizers
            self.opt_classic.step()
            if self.opt_quantum:
                self.opt_quantum.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            avg_loss = epoch_loss / (i + 1)
            
            # Рендерим полоску и обновляем прогресс-бар
            l_bar = Utils.get_loss_bar(current_loss)
            pbar.set_postfix_str(f"Loss: {current_loss:.4f} {l_bar} Avg: {avg_loss:.4f}")
            
        return avg_loss

    def validate(self, loader) -> float:
        """
        Validate the model on validation set.

        Args:
            loader: DataLoader for validation data.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]
                _, _, _, _, targets = batch
                preds = self.model(batch).squeeze()
                val_loss += self.criterion(preds, targets).item()
        return val_loss / len(loader)

    def train(self, train_loader, val_loader, exp_dir) -> Tuple[int, float]:
        """
        Run the complete training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
        """
        best_val_r = -1.0
        best_epoch = 0
        self.history = {
            'train_loss': [], 'val_rmse': [], 'val_pearson': [], 
            'val_ci': [], 'best_y_true': None, 'best_y_pred': None
            }
        
        for epoch in range(self.train_cfg['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            log_info(f"Epoch {epoch+1}/{self.train_cfg['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", stage="PROGRESS")

            rmse, r_val, ci_val, preds, targets = self.evaluator.evaluate(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_rmse'].append(rmse)
            self.history['val_pearson'].append(r_val)
            self.history['val_ci'].append(ci_val)
            if r_val >= max(self.history['val_pearson']):
                self.history['best_y_true'] = targets.tolist()
                self.history['best_y_pred'] = preds.tolist()

            with open(f"{exp_dir}/history.json", 'w') as f:
                json.dump(self.history, f, indent=4)
            torch.save(self.model.state_dict(), f"{exp_dir}/model_epoch_{epoch}.pt")

            log_info(f"Valid: RMSE {rmse:.4f} | R {r_val:.4f} | CI {ci_val:.4f}", stage="PROGRESS")
            log_info("-" * 60, stage="PROGRESS")

            if r_val > best_val_r:
                best_val_r = r_val
                best_epoch = (epoch + 1)
                torch.save(self.model.state_dict(), f"{exp_dir}/best_model.pt")
                log_info(f"New best R: {best_val_r:.4f} (Saved to best_model.pt)", stage="TRAINER")
            
            self.step_schedulers(val_loss)

        return best_epoch, best_val_r

    def test(self, test_loader, exp_dir, best_epoch, show_plots=False, save_plots=True):
        if hasattr(self, 'history'):
            self.evaluator.plot_history(exp_dir, self.history, show=show_plots, save=save_plots)
        else:
            log_info("No training history available for plotting.", stage="TEST")

        log_info("FINAL TEST (CORE SET)", stage="TEST")
        # Подгружаем веса лучшей эпохи (в идеале нужно написать логику загрузки лучшего .pt,
        # но пока протестируем на весах последней эпохи)

        best_model_path = f"{exp_dir}/best_model.pt"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            log_info(f"Успешно загружены веса лучшей эпохи {best_epoch} из {best_model_path}", stage="TRAINER")

        test_rmse, test_r, test_ci, _, _ = self.evaluator.evaluate(test_loader)
        log_info(f"FINAL TEST -> RMSE: {test_rmse:.4f} | Pearson R: {test_r:.4f} | CI: {test_ci:.4f}", stage="TEST")
        
        # Сохраняем результаты теста
        with open(f"{exp_dir}/test_results.json", 'w') as f:
            json.dump({
                "RMSE": test_rmse,
                "Pearson_R": test_r,
                "CI": test_ci
            }, f, indent=4)
