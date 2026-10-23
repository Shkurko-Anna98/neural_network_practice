"""
Главный файл для командной строки (CLI).
Здесь обрабатываются команды prepare и doctor.
"""
import argparse
import logging
import sys
import subprocess
import platform
from pathlib import Path

from .config import Config
from .logging_config import setup_logging
from .scanner import DatasetScanner

logger = logging.getLogger(__name__)

def run_prepare(args):
    """Обработка команды prepare (подготовка данных)."""
    logger.info("Запущена команда prepare...")
    
    # Берем путь из аргументов, если не указан - из конфига
    data_path = Path(args.path) if args.path else Config.DATA_DIR
    out_path = Path(args.output) if args.output else Config.OUTPUT_FILE
    
    # Расширения. Если пользователь не указал, берем картинки и CSV по умолчанию
    extensions = args.ext.split(',') if args.ext else ['.jpg', '.png', '.csv']
    
    logger.info(f"Папка для сканирования: {data_path}")
    logger.info(f"Ищем файлы с расширениями: {extensions}")
    logger.info(f"Сохраняем результат в: {out_path}")

    scanner = DatasetScanner(directory=data_path, extensions=extensions, output_file=out_path)
    scanner.scan_and_process()
    logger.info("Команда prepare завершена.")

def run_doctor(args):
    """Обработка команды doctor (проверка окружения)."""
    print("--- ПРОВЕРКА ОКРУЖЕНИЯ ---")
    
    # 1. Версия Python
    print(f"Версия Python: {platform.python_version()}")
    
    # 2. Проверка библиотек
    print("\nПроверка библиотек:")
    libraries = ['numpy', 'PIL', 'dotenv']
    for lib in libraries:
        try:
            __import__(lib)
            print(f"  [+] {lib} - установлена")
        except ImportError:
            print(f"  [-] {lib} - НЕ НАЙДЕНА!")

    # 3. Проверка CUDA (видеокарты NVIDIA)
    print("\nПроверка CUDA (видеокарта):")
    try:
        # Пытаемся вызвать команду nvidia-smi
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print("  [+] Видеокарта NVIDIA найдена. CUDA доступна.")
        # Можно вывести первую строчку с названием видеокарты
        print(f"      {result.stdout.splitlines()[2]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  [-] nvidia-smi не найдена. CUDA недоступна (или нет видеокарты NVIDIA).")

    # 4. Текущая директория
    print(f"\nТекущая рабочая директория: {Path.cwd()}")
    print("--- ПРОВЕРКА ЗАВЕРШЕНА ---")

def main():
    """Главная функция, которая разбирает аргументы командной строки."""
    setup_logging()
    
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="CLI утилита для подготовки данных")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # Настройка команды prepare
    parser_prepare = subparsers.add_parser("prepare", help="Собрать данные в .npy файл")
    parser_prepare.add_argument("--path", type=str, help="Путь к папке с данными")
    parser_prepare.add_argument("--ext", type=str, help="Расширения файлов через запятую (например, .jpg,.png)")
    parser_prepare.add_argument("--output", type=str, help="Путь к выходному файлу .npy")
    parser_prepare.set_defaults(func=run_prepare)

    # Настройка команды doctor
    parser_doctor = subparsers.add_parser("doctor", help="Проверить окружение")
    parser_doctor.set_defaults(func=run_doctor)

    # Разбираем, что ввел пользователь
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # Запускаем нужную функцию
    args.func(args)

if __name__ == "__main__":
    main()