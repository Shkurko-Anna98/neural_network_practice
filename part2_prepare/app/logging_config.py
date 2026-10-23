"""
Настройка логов для программы.
"""
import logging
import os
import sys

def setup_logging():
    """
    Включает логирование. Уровень берем из окружения, если нет - ставим INFO.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Настраиваем, чтобы логи шли и в консоль, и в файл
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/prepare.log", mode='a', encoding='utf-8')
        ]
    )
    logging.info("Логирование включено.")