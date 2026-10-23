"""
Сканируем папки и собираем все файлы в один массив.
"""
import logging
import time
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class DatasetScanner:
    """Класс, который ищет файлы и делает из них датасет."""

    def __init__(self, directory: Path, extensions: list, output_file: Path, image_size=(64, 64)):
        """Принимает папку, список расширений и куда сохранять."""
        self.directory = directory
        self.extensions = [ext.lower() for ext in extensions]
        self.output_file = output_file
        self.image_size = image_size

    def _process_image(self, file_path: Path) -> np.ndarray:
        """Открывает картинку, меняет размер, нормализует и делает массив."""
        with Image.open(file_path) as img:
            # Конвертируем в RGB на всякий случай (если картинка черно-белая)
            img = img.convert('RGB')
            # Приводим все картинки к одному размеру
            img = img.resize(self.image_size)
            arr = np.array(img, dtype=np.float32)
            # Нормализация (пиксели от 0 до 255 делим на 255)
            arr = arr / 255.0
            return arr

    def _process_csv(self, file_path: Path) -> np.ndarray:
        """Читает CSV файл как таблицу с числами."""
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=np.float32)
        return data

    def scan_and_process(self) -> None:
        """Обходит папки, обрабатывает файлы и сохраняет результат."""
        start_time = time.time()
        logger.info(f"Начинаем сканировать: {self.directory}")
        
        if not self.directory.exists():
            logger.error(f"Папка {self.directory} не найдена!")
            return

        dataset = []
        file_count = 0

        # Ищем файлы во всех вложенных папках
        for file_path in self.directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.extensions:
                try:
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        arr = self._process_image(file_path)
                        dataset.append(arr)
                    elif file_path.suffix.lower() == '.csv':
                        arr = self._process_csv(file_path)
                        dataset.append(arr)
                    
                    file_count += 1
                    logger.debug(f"Готово: {file_path.name}")
                except Exception as e:
                    logger.error(f"Ошибка с файлом {file_path.name}: {e}")

        if file_count == 0:
            logger.warning("Не нашли файлы с такими расширениями.")
            return

        # Склеиваем все массивы в один
        try:
            final_dataset = np.stack(dataset)
            logger.info(f"Обработано файлов: {file_count}")
            logger.info(f"Размер массива: {final_dataset.shape}")
            
            # Создаем папку для сохранения, если её еще нет
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(self.output_file, final_dataset)
            logger.info(f"Сохранили в: {self.output_file}")
            
        except Exception as e:
            logger.error(f"Ошибка при склейке данных (размеры не совпадают?): {e}")

        end_time = time.time()
        logger.info(f"Время работы: {end_time - start_time:.2f} сек.")