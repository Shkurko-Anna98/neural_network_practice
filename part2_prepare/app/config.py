"""
Файл с настройками. Читаем пути из .env.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Класс, где хранятся наши пути и настройки."""
    
    DATA_DIR = Path(os.getenv("DATA_DIR", "data/input"))
    OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "data/output/dataset.npy"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def print_config(cls):
        """Печатаем настройки, чтобы проверить, что все загрузилось."""
        print(f"Текущие настройки:")
        print(f"  Папка с данными: {cls.DATA_DIR}")
        print(f"  Файл результата: {cls.OUTPUT_FILE}")
        print(f"  Уровень логов: {cls.LOG_LEVEL}")