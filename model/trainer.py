# model/trainer.py

import torch
# import torch.nn as nn
from tqdm import tqdm
from model.model import UniversalHybridSlotModel
from utils import Utils
from loss_functions.loss_functions import get_loss_function
from logger import *
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

        log_debug("--- FULL MODEL PARAMETERS SCAN ---", stage="DEBUG")
        all_params = list(model.named_parameters())
        for name, p in all_params:
            log_debug(f"Name: {name} | Shape: {list(p.shape)} | RequiresGrad: {p.requires_grad}", stage="OPTIMIZER")

        q_keys = ["qlayer", "final_layer"]
        quantum_params = [p for n, p in model.named_parameters() if any(k in n for k in q_keys)]
        classic_params = [p for n, p in model.named_parameters() if not any(k in n for k in q_keys)]

        # 2. Initialize optimizers from config
        c_opt_cfg = self.train_cfg['optimizers']['classic']
        self.opt_classic = getattr(torch.optim, c_opt_cfg['type'])(classic_params, **c_opt_cfg['params'])
        log_info(f"Classic: {c_opt_cfg['type']} with {sum(p.numel() for p in classic_params)}"
                 f" parameters in {len(classic_params)} tensors", stage="OPTIMIZER")

        q_opt_cfg = self.train_cfg['optimizers']['quantum']
        if len(quantum_params) > 0:
            self.opt_quantum = getattr(torch.optim, q_opt_cfg['type'])(quantum_params, **q_opt_cfg['params'])
            log_info(f"Quantum: {q_opt_cfg['type']} with {sum(p.numel() for p in quantum_params)}"
                     f" parameters in {len(quantum_params)} tensors", stage="OPTIMIZER")
        else:
            self.opt_quantum = None
            log_warn(f"Quantum: No quantum parameters found, optimizer disabled", stage="OPTIMIZER")

        self.sched_classic = self._build_scheduler(self.opt_classic, c_opt_cfg.get('scheduler'))
        self.sched_quantum = self._build_scheduler(self.opt_quantum, q_opt_cfg.get('scheduler')) if self.opt_quantum else None

        # 3. Loss function
        self.criterion = get_loss_function(config['training'])

        es_cfg = config['training']['early_stopping']
        self.es_enabled = es_cfg['enabled']
        self.monitors = es_cfg['monitors']
        self.primary_metric = es_cfg['primary_monitor']
        self.primary_metric_mode = es_cfg['monitors'][self.primary_metric]
        if self.primary_metric_mode == 'ignore':
            log_error(f"Primary metric {self.primary_metric_mode} can't be ignored", stage="METRICS")
        log_info(f"Primary metric: {self.primary_metric} with mode {self.primary_metric_mode}", stage="METRICS")
        self.best_scores = {self.primary_metric: float('-inf') if self.primary_metric_mode == 'max' else float('inf')}
        if self.es_enabled:
            self.es_patience = es_cfg['patience']
            self.early_stop = False

            for k, v in self.monitors.items():
                if v == 'max':
                    self.best_scores[k] = float('-inf')
                elif v == 'min':
                    self.best_scores[k] = float('inf')
                elif v == 'ignore':
                    pass
                else:
                    log_error(f"Unknown mode {v} on metrics {k} monitoring", stage="METRICS")
            log_info(f"Early stopping enabled with patience {self.es_patience}", stage="METRICS")
            log_info(f"Monitors: {self.monitors}", stage="METRICS")
        else:
            log_info("Early stopping disabled", stage="TRAINER")

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

    @staticmethod
    def _collect_tensor_stats(params):
        total_num = 0
        total_sum = 0.0
        total_sum_sq = 0.0
        min_val = float('inf')
        max_val = float('-inf')

        for p in params:
            if p is None:
                continue
            tensor = p.detach()
            if tensor.numel() == 0:
                continue
            flat = tensor.view(-1)
            total_num += flat.numel()
            total_sum += float(flat.sum().item())
            total_sum_sq += float(flat.pow(2).sum().item())
            min_val = min(min_val, float(flat.min().item()))
            max_val = max(max_val, float(flat.max().item()))

        if total_num == 0:
            return None
        mean = total_sum / total_num
        variance = max(total_sum_sq / total_num - mean ** 2, 0.0)
        std = variance ** 0.5
        return {
            'count': total_num,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
        }

    @staticmethod
    def _optimizer_lrs(optimizer):
        if optimizer is None:
            return None
        return [float(param_group.get('lr', 0.0)) for param_group in optimizer.param_groups]

    def _log_epoch_start(self, epoch_id: int, total_epochs: int, progress: float) -> None:
        classic_lr = self._optimizer_lrs(self.opt_classic)
        quantum_lr = self._optimizer_lrs(self.opt_quantum)

        q_keys = ["qlayer", "final_layer"]
        q_params = [p for n, p in self.model.named_parameters() if any(k in n for k in q_keys)]
        q_param_count = len(q_params)
        q_stats = self._collect_tensor_stats(q_params)

        log_info(
            f"[EPOCH {epoch_id}/{total_epochs}] progress={progress:.3f} classic_lr={classic_lr}"
            f" quantum_lr={quantum_lr} ",
            stage="TRAINER"
        )
        if q_stats is not None:
            log_info(
                f"[EPOCH {epoch_id}/{total_epochs}] quantum_params={sum(p.numel() for p in q_params)}"
                f" in {q_param_count} tensors mean={q_stats['mean']:.6f} "
                f"std={q_stats['std']:.6f} min={q_stats['min']:.6f} max={q_stats['max']:.6f}",
                stage="TRAINER"
            )

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

    def train_epoch(self, loader, progress: float = 0.0) -> float:
        """
        Train for one epoch.

        Args:
            loader: DataLoader for training data.
            progress: Schedule progress in [0, 1] for the quantum encoder.

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
            preds = self.model(batch, progress=progress).squeeze()
            loss = self.criterion(preds, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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

    def validate(self, loader, progress: float = 1.0) -> float:
        """
        Validate the model on validation set.

        Args:
            loader: DataLoader for validation data.
            progress: Schedule progress in [0, 1] for the quantum encoder.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]
                _, _, _, _, targets = batch
                preds = self.model(batch, progress=progress).squeeze()
                val_loss += self.criterion(preds, targets).item()
        return val_loss / len(loader)

    def train(self, train_loader, val_loader, exp_dir, save_only_best_epoch=True) -> Tuple[int, float]:
        """
        Run the complete training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
        """
        plot_every_n_epochs = self.train_cfg.get('plot_every_n_epochs', 10)
        # best_val_r = -1.0
        best_epoch = 0
        self.history = {
            'train_loss': [], 'val_rmse': [], 'val_pearson': [], 
            'val_ci': [], 'best_y_true': None, 'best_y_pred': None
            }
        if self.es_enabled:
            self.es_counter = 0
        
        total_number_of_epochs = self.train_cfg['epochs']
        log_info("-" * 75, stage="TRAINER")
        for epoch in range(total_number_of_epochs):
            epoch_id = epoch + 1
            progress = epoch / max(1, total_number_of_epochs - 1)
            self._log_epoch_start(epoch_id, total_number_of_epochs, progress)
            train_loss = self.train_epoch(train_loader, progress=progress)
            val_loss = self.validate(val_loader, progress=progress)
            
            log_info(f"[EPOCH {epoch_id}/{total_number_of_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", stage="TRAINER")

            _, r_val, ci_val, preds, targets = self.evaluator.evaluate(val_loader, progress=progress)

            if self.config['dataset']['stats'] is not None:
                stats = self.config['dataset']['stats']
            else:
                raise ValueError("Data are denormalized.")
            preds_denorm = Utils.denormalize(preds, stats)
            targets_denorm = Utils.denormalize(targets, stats)
            rmse_denorm = Utils.calculate_rmse(targets_denorm, preds_denorm)
            log_debug(f"{type(train_loss)}, {type(rmse_denorm)}, {type(r_val)}, {type(ci_val)}", stage="TRAINER")
            log_debug(f"{type(preds_denorm)}, {type(targets_denorm)}, {type(preds_denorm[0])}, {type(targets_denorm[0])}", stage="TRAINER")
            log_debug(f"{train_loss}, {rmse_denorm}, {r_val}, {ci_val}", stage="TRAINER")
            log_debug(f"{preds_denorm}, {targets_denorm}, {preds_denorm[0]}, {targets_denorm[0]}", stage="TRAINER")
            self.history['train_loss'].append(float(train_loss))
            self.history['val_rmse'].append(float(rmse_denorm))
            self.history['val_pearson'].append(float(r_val))
            self.history['val_ci'].append(float(ci_val))
            log_info(f"[EPOCH {epoch_id}/{total_number_of_epochs}] Valid: RMSE {rmse_denorm:.4f} | R {r_val:.4f} | CI {ci_val:.4f}", stage="TRAINER")

            current_metrics = {
                "val_pearson": r_val,
                "val_rmse": rmse_denorm,
                "train_loss": train_loss,
                "val_ci": ci_val
            }

            improved_any = False
            improved_primary = False

            for metric, mode in self.monitors.items():
                if mode == 'ignore': continue
                val = current_metrics.get(metric)
                if val is None: continue
                if (mode == 'max' and val > self.best_scores[metric]) or \
                    (mode == 'min' and val < self.best_scores[metric]):

                    self.best_scores[metric] = val
                    improved_any = True
                    log_info(f"{"Primary m" if self.primary_metric == metric else "M"}etric"
                    f" {metric} improved: {val:.4f}", stage="TRAINER")

                    if metric == self.primary_metric:
                        improved_primary = True

            if improved_primary:
                best_epoch = epoch_id
                torch.save(self.model.state_dict(), f"{exp_dir}/best_model.pt")
                self.history['best_y_true'] = targets_denorm.tolist()
                self.history['best_y_pred'] = preds_denorm.tolist()
                log_info(f"New best for primary metric {self.primary_metric}:"
                f" {self.best_scores[self.primary_metric]:.4f} (Saved to best_model.pt)",
                stage="TRAINER")
            if self.es_enabled:
                if improved_any:
                    self.es_counter = 0
                else:
                    self.es_counter += 1
                    log_info(f"EarlyStopping counter: {self.es_counter}/{self.es_patience}",
                                stage="TRAINER")
                    if self.es_counter >= self.es_patience:
                        log_info(f"[EPOCH {epoch_id}/{total_number_of_epochs}]"
                        f" Early stopping triggered", stage="TRAINER")
                        self.early_stop = True

            for k in ['train_loss', 'val_rmse', 'val_pearson', 'val_ci']:
                data = self.history[k]
                log_debug(f"Key: {k}, Length: {len(data)}, Types: {[type(x) for x in data]}", stage="DEBUG_PLOT")

            with open(f"{exp_dir}/history.json", 'w') as f:
                json.dump(self.history, f, indent=4)

            if not save_only_best_epoch:
                torch.save(self.model.state_dict(), f"{exp_dir}/model_epoch_{epoch_id}.pt")

            log_info("-" * 75, stage="TRAINER")
            if self.early_stop:
                break

            self.step_schedulers(val_loss)

            # if it is the last epoch, the runner anyway will draw the results
            if (epoch_id % plot_every_n_epochs == 0) and (epoch_id != total_number_of_epochs):
                console_plots(self.history, side_by_side=True, stage="TRAINER")

        log_info("-" * 75, stage="TRAINER")
        return best_epoch, self.best_scores[self.primary_metric]

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
            log_info(f"Weights for the best model (epoch {best_epoch}) loaded from {best_model_path}", stage="TEST")

        test_rmse, test_r, test_ci, _, _ = self.evaluator.evaluate(test_loader, progress=1.0)
        log_info(f"FINAL TEST -> RMSE: {test_rmse:.4f} | Pearson R: {test_r:.4f} | CI: {test_ci:.4f}", stage="TEST")
        
        # Сохраняем результаты теста
        with open(f"{exp_dir}/test_results.json", 'w') as f:
            json.dump({
                "RMSE": test_rmse,
                "Pearson_R": test_r,
                "CI": test_ci
            }, f, indent=4)
