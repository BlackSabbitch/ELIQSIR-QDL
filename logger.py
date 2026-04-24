# logger.py

import logging

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
