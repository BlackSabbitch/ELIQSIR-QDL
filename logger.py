# logger.py

import logging
import io
from contextlib import redirect_stdout
from uniplot import plot as uplot

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
    formatter = StageFormatter('[%(levelname)s][%(stage)s] - %(message)s')
    fh.setFormatter(formatter)
    
    # Добавляем к существующему логгеру
    logger.addHandler(fh)

def get_ascii_plot(data, title, width=50, height=10, lines=False):
    """Генерирует график и ГАРАНТИРОВАННО возвращает его как список строк"""
    if data is None or (isinstance(data, list) and len(data) == 0):
        return ["No data available"]
        
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            # Передаем x и y для scatter или просто y для обычного
            if isinstance(data, list) and len(data) == 2 and isinstance(data[0], (list, np.ndarray)):
                # Это Scatter Plot (Actual vs Predicted)
                uplot(xs=data[0], ys=data[1], title=title, width=width, height=height, lines=lines, color=False)
            else:
                # Это обычный график (Loss, RMSE)
                uplot(ys=data, title=title, width=width, height=height, lines=lines, color=False)
        
        result = buf.getvalue().splitlines()
        return result if result else ["Plot generation failed"]
    except Exception as e:
        return [f"Error drawing plot: {str(e)}"]

def log_side_by_side(data_left, title_left, data_right, title_right, lines_left=False, lines_right=False, is_scatter=False):
    # Гарантируем, что получаем списки строк

    # left_lines = list(get_ascii_plot(data_left, title_left, width=45, lines=not is_scatter))
    left_lines = list(get_ascii_plot(data_left, title_left, width=45, lines=lines_left))
    right_lines = list(get_ascii_plot(data_right, title_right, width=45, lines=lines_right))

    max_len = max(len(left_lines), len(right_lines))
    
    # Теперь конкатенация списков (list + list) сработает корректно
    left_lines += [""] * (max_len - len(left_lines))
    right_lines += [""] * (max_len - len(right_lines))

    combined = [] # Начинаем со списка строк
    for left, right in zip(left_lines, right_lines):
        # ljust(55) выравнивает колонку, чтобы вертикальная черта была ровной
        combined.append(f"{left.ljust(50)} \t\t {right}")
    
    # Объединяем всё в одну большую строку для логгера
    final_output = "\n" + "\n".join(combined)
    
    return final_output

import numpy as np

def log_residuals_hist(y_true, y_pred):
    errors = np.array(y_pred) - np.array(y_true)
    counts, bins = np.histogram(errors, bins=20)
    # Рисуем центры бинов против их частоты
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_chart = get_ascii_plot(counts, title="Error Distribution (Residuals)", width=100)
    log_info(f"Residuals Histogram:\n" + "\n".join(hist_chart), stage="SUMMARY")
    return hist_chart
