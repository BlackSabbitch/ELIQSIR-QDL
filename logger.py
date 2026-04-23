# logger.py

import logging


# Custom formatter for prefixed logging
class PrefixedFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'prefix'):
            record.msg = f"[{record.prefix}] {record.msg}"
        return super().format(record)

# Setup logging with custom formatter
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = PrefixedFormatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Convenience functions for custom prefixes
def log_error_type_1(msg: str) -> None:
    logger.info(msg, extra={'prefix': 'ERROR TYPE 1'})

def log_error_type_2(msg: str) -> None:
    logger.info(msg, extra={'prefix': 'ERROR TYPE 2'})

def log_info_type_3(msg: str) -> None:
    logger.info(msg, extra={'prefix': 'INFORMATION TYPE 3'})
