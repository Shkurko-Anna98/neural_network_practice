"""
Реализация паттерна Producer-Consumer с потоками и процессами.
"""
import time
import threading
import queue
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np

from .augment import apply_augmentation
from .performance import save_performance_report

# Глобальная переменная для подсчета файлов
processed_counter = 0
lock = threading.Lock() # Замок для защиты счетчика

def producer(directory, file_queue):
    """Ищет файлы и кладет их пути в очередь."""
    files = list(Path(directory).rglob("*.jpg")) + list(Path(directory).rglob("*.png"))
    for file in files:
        file_queue.put(file)
    
    # Кладем "стоп-сигналы" для каждого потока-потребителя (у нас их 3)
    for _ in range(3):
        file_queue.put(None)

def consumer(file_queue, array_queue):
    """Берет путь из очереди, читает файл и кладет массив в другую очередь."""
    global processed_counter
    while True:
        file_path = file_queue.get()
        if file_path is None: # Если пришел стоп-сигнал, выходим
            break
            
        try:
            # Открываем и нормализуем картинку
            with Image.open(file_path) as img:
                img = img.convert('RGB').resize((64, 64))
                arr = np.array(img, dtype=np.float32) / 255.0
                array_queue.put(arr)
            
            # Увеличиваем счетчик с блокировкой
            with lock:
                processed_counter += 1
                print(f"[Поток] Прочитан файл: {file_path.name} (Всего: {processed_counter})")
        except Exception as e:
            print(f"Ошибка чтения {file_path.name}: {e}")
        
        file_queue.task_done()

def run_sequential(directory):
    """Последовательная версия (для сравнения времени)."""
    print("\n--- Последовательная обработка ---")
    start_time = time.time()
    count = 0
    for p in Path(directory).rglob("*.jpg"):
        with Image.open(p) as img:
            img = img.convert('RGB').resize((64, 64))
            arr = np.array(img, dtype=np.float32) / 255.0
            apply_augmentation(arr)
            count += 1
    end_time = time.time()
    print(f"Последовательно обработано: {count} файлов.")
    return end_time - start_time

def run_parallel(directory):
    """Параллельная версия (потоки для чтения, процессы для аугментации)."""
    print("\n--- Параллельная обработка ---")
    global processed_counter
    processed_counter = 0
    
    start_time = time.time()
    
    file_queue = queue.Queue()
    array_queue = queue.Queue()

    # 1. Запускаем поток-производитель
    prod_thread = threading.Thread(target=producer, args=(directory, file_queue))
    prod_thread.start()

    # 2. Запускаем потоки-потребители
    threads = []
    for _ in range(3):
        t = threading.Thread(target=consumer, args=(file_queue, array_queue))
        t.start()
        threads.append(t)

    # Ждем, пока все потоки закончат работу
    prod_thread.join()
    for t in threads:
        t.join()

    # 3. Забираем все массивы из очереди для обработки процессами
    arrays_to_process = []
    while not array_queue.empty():
        arrays_to_process.append(array_queue.get())

    print(f"[Потоки] Закончили чтение. Массивов в памяти: {len(arrays_to_process)}")

    # 4. Запускаем пул процессов для аугментации
    if arrays_to_process:
        with multiprocessing.Pool() as pool:
            # apply_async берет функцию и аргументы, и делает это в отдельном процессе
            results = [pool.apply_async(apply_augmentation, (arr,)) for arr in arrays_to_process]
            # Ждем завершения всех процессов
            augmented_data = [res.get() for res in results]
    
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    # Папка с нашими картинками, которые мы скачали для второй части
    test_dir = Path("data/input/images")
    
    if not test_dir.exists():
        print("Сначала создайте папку data/input/images и положите туда картинки!")
    else:
        # Замеряем последовательную версию
        seq_time = run_sequential(test_dir)
        
        # Замеряем параллельную версию
        par_time = run_parallel(test_dir)
        
        # Сохраняем отчет
        save_performance_report(seq_time, par_time)