# logger.py

import logging
import io
import numpy as np
from contextlib import redirect_stdout
from uniplot import plot as uplot

DEFAULT_WIDTH_SINGLE = 80
DEFAULT_HEIGHT_SINGLE = 25
DEFAULT_WIDTH_SIDE = 50
DEFAULT_HEIGHT_SIDE = 15
# 50 (width) + 10 (labels/padding) = 60
LEFT_COLUMN_TOTAL_WIDTH = DEFAULT_WIDTH_SIDE + 10


class StageFormatter(logging.Formatter):
    def format(self, record):
        # Если префикс не передан, ставим пустую заглушку, 
        # чтобы форматтер не выкинул ошибку
        if not hasattr(record, 'stage'):
            record.stage = "GENERAL"
        return super().format(record)

# Настройка
logger = logging.getLogger("AppCore")
handler = logging.StreamHandler()

# Вот здесь задаем твою схему [LEVEL][STAGE]
formatter = StageFormatter('[%(levelname)s][%(stage)s] %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def _log(msg, stage, level):
    logger.log(level, msg, extra={'stage': stage})

# Публичный API твоего логгера
def log_info(msg, stage="GENERAL"):
    _log(msg, stage, logging.INFO)

def log_warn(msg, stage="GENERAL"):
    _log(msg, stage, logging.WARNING)

def log_error(msg, stage="GENERAL"):
    _log(msg, stage, logging.ERROR)

def log_debug(msg, stage="GENERAL"):
    _log(msg, stage, logging.DEBUG)

def setup_file_logging(log_path):
    # Создаем обработчик для файла
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # Применяем тот же форматтер
    formatter = StageFormatter('[%(levelname)s][%(stage)s] %(message)s')
    fh.setFormatter(formatter)
    
    # Добавляем к существующему логгеру
    logger.addHandler(fh)

def get_ascii_plot(data, title,
                   width=DEFAULT_WIDTH_SINGLE,
                   height=DEFAULT_HEIGHT_SINGLE,
                   lines=False,
                   force_diagonal=False,
                   stage="SUMMARY"):
    if data is None: return ["No data"]
    
    buf = io.StringIO()
    # Словарь аргументов для uplot
    uplot_args = {
        "title": title,
        "width": width,
        "height": height,
        "color": False,
        "lines": lines
    }

    # 1. Логика для Scatter Plot (Actual vs Predicted)
    if isinstance(data, list) and len(data) == 2 and isinstance(data[0], (list, np.ndarray)):
        xs_raw, ys_raw = np.array(data[0]), np.array(data[1])
        
        if force_diagonal:
            low = min(xs_raw.min(), ys_raw.min())
            high = max(xs_raw.max(), ys_raw.max())
            
            # Устанавливаем жесткие границы квадрата
            uplot_args.update({
                "x_min": low, "x_max": high,
                "y_min": low, "y_max": high,
                "xs": [xs_raw, np.array([low, high])],
                "ys": [ys_raw, np.array([low, high])],
                "lines": [False, True] # Точки для данных, линия для диагонали
            })
        else:
            uplot_args.update({"xs": xs_raw, "ys": ys_raw})
            
    # 2. Логика для обычных графиков (Loss, RMSE и Histogram)
    else:
        # Если это гистограмма, пришедшая как [bins, counts]
        if isinstance(data, list) and len(data) == 2 and isinstance(data[0], (list, np.ndarray)):
             uplot_args.update({"xs": data[0], "ys": data[1]})
        else:
             uplot_args.update({"ys": data})

    try:
        with redirect_stdout(buf):
            uplot(**uplot_args) # Распаковываем только нужные аргументы
        return buf.getvalue().splitlines()
    except Exception as e:
        log_error(f"{title} Plot Error: {e}", stage=stage)
        return [f"{title} Plot Error: {e}"]

def get_residuals_hist_data(y_true, y_pred, bins=20):
    """Готовит данные для гистограммы"""
    errors = np.array(y_pred) - np.array(y_true)
    counts, bin_edges = np.histogram(errors, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def log_side_by_side(data_left, title_left, data_right, title_right, 
                     is_scatter=False, is_hist=False, stage="SUMMARY"):
    # Если слева scatter, включаем диагональ
    left_lines = get_ascii_plot(data_left, title_left,
                                width=DEFAULT_WIDTH_SIDE, height=DEFAULT_HEIGHT_SIDE,
                                force_diagonal=is_scatter, stage=stage)
    
    # Если справа гистограмма, рисуем ее линиями
    right_lines = get_ascii_plot(data_right, title_right,
                                 width=DEFAULT_WIDTH_SIDE, height=DEFAULT_HEIGHT_SIDE,
                                 lines=is_hist, stage=stage)

    max_len = max(len(left_lines), len(right_lines))
    left_lines += [""] * (max_len - len(left_lines))
    right_lines += [""] * (max_len - len(right_lines))

    max_w_left = max(len(line) for line in left_lines) if left_lines else DEFAULT_WIDTH_SIDE

    combined = [""]
    separator = " " * 4
    for left, right in zip(left_lines, right_lines):
        left_padded = left.ljust(LEFT_COLUMN_TOTAL_WIDTH)
        combined.append(f"{left_padded}{separator}{right}")
        # combined.append(f"{left.ljust(DEFAULT_WIDTH_SIDE + 5)}{separator}{right}")
    
    return "\n".join(combined)

def console_plots(trainer_history, side_by_side=True, stage="SUMMARY"):
    if side_by_side:
        log_info("Generating Multi-column ASCII Dashboard...", stage=stage)

        dashboard_loss_rmse = log_side_by_side(
            trainer_history["train_loss"], "Learning Curve (Loss)",
            trainer_history["val_rmse"], "Validation RMSE"
        )
        log_info(dashboard_loss_rmse, stage=stage)

        dashboard_r_ci = log_side_by_side(
            trainer_history["val_pearson"], "Correlation (Pearson R)",
            trainer_history["val_ci"], "Ranking Accuracy (CI)"
        )
        log_info(dashboard_r_ci, stage=stage)

        y_true = np.array(trainer_history['best_y_true'])
        y_pred = np.array(trainer_history['best_y_pred'])

        hist_x, hist_y = get_residuals_hist_data(y_true, y_pred)

        dashboard_final = log_side_by_side(
            [y_true, y_pred], "Actual vs Predicted",   # Left: Scatter
            [hist_x, hist_y], "Residuals Distribution", # Right: Histogram (как XY график)
            is_scatter=True,
            is_hist=True,
            stage=stage
        )
        
        log_info("Final Performance Analytics:" + dashboard_final, stage=stage)

    else:
        loss_chart = get_ascii_plot(trainer_history["train_loss"], title="Learning Curve (Loss)")
        log_info(f"Loss Curve:\n" + "\n".join(loss_chart), stage=stage)

        rmse_chart = get_ascii_plot(trainer_history["val_rmse"], title="Validation RMSE")
        log_info(f"RMSE Curve:\n" + "\n".join(rmse_chart), stage=stage)

        r_chart = get_ascii_plot(trainer_history["val_pearson"], title="Correlation (Pearson R)")
        log_info(f"Pearson (R) Curve:\n" + "\n".join(r_chart), stage=stage)

        ci_chart = get_ascii_plot(trainer_history["val_ci"], title="Ranking Accuracy (CI)")
        log_info(f"Concordancy Index Curve:\n" + "\n".join(ci_chart), stage=stage)

        y_true = trainer_history['best_y_true']
        y_pred = trainer_history['best_y_pred']

        act_vs_pred_chart = get_ascii_plot([y_true, y_pred], title="Predicted = f(Actual)", force_diagonal=True)
        log_info(f"Actual vs Predicted Scatter Plot:\n" + "\n".join(act_vs_pred_chart), stage=stage)

        hist_x, hist_y = get_residuals_hist_data(y_true, y_pred)

        resid_chart = get_ascii_plot([hist_x, hist_y], title="Residuals Distribution", lines=True)
        log_info(f"Errors Distribution (Residuals):\n" + "\n".join(resid_chart), stage=stage)
