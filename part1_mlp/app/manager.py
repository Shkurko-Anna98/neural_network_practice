"""
Модуль для управления данными: загрузка из CSV, нормализация и разбиение на выборки.
"""
import csv
import numpy as np
import os
from .exceptions import FileLoadError, MismatchedDataError

class DatasetManager:
    """
    Класс для загрузки, нормализации и подготовки датасета.
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # Сохраняем параметры нормализации, чтобы потом применить их к новым данным
        self.min_vals = None
        self.max_vals = None

    def load_csv(self, filepath):
        """
        Загружает данные из CSV-файла.
        Первая строка считается заголовками, последний столбец — целевой переменной.
        """
        if not os.path.exists(filepath):
            raise FileLoadError(f"Файл не найден по пути: {filepath}")

        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Читаем первую строку
                
                if len(headers) < 2:
                    raise MismatchedDataError("В CSV должно быть как минимум 2 столбца (признаки и цель).")

                data = []
                for row in reader:
                    # Пропускаем пустые строки
                    if not row:
                        continue
                    # Преобразуем строки в числа
                    data.append([float(x) for x in row])

            data = np.array(data)
            
            # Последний столбец - это y (целевая переменная), остальные - X (признаки)
            self.X = data[:, :-1]
            self.y = data[:, -1]
            
            self.feature_names = headers[:-1]
            self.target_name = headers[-1]
            
            print(f"Данные успешно загружены. Признаков: {self.X.shape[1]}, примеров: {self.X.shape[0]}")
            
        except Exception as e:
            raise FileLoadError(f"Ошибка при чтении CSV файла: {e}")

    def normalize(self, method='minmax'):
        """
        Нормализует признаки.
        :param method: 'minmax' (от 0 до 1) или 'standard' (среднее 0, отклонение 1)
        """
        if self.X is None:
            raise MismatchedDataError("Данные не загружены. Сначала вызовите load_csv().")

        if method == 'minmax':
            self.min_vals = np.min(self.X, axis=0)
            self.max_vals = np.max(self.X, axis=0)
            
            # Избегаем деления на ноль, если все значения в столбце одинаковые
            range_vals = self.max_vals - self.min_vals
            range_vals[range_vals == 0] = 1 
            
            self.X = (self.X - self.min_vals) / range_vals
            
        elif method == 'standard':
            mean = np.mean(self.X, axis=0)
            std = np.std(self.X, axis=0)
            std[std == 0] = 1
            self.X = (self.X - mean) / std
            
        else:
            raise ValueError("Неизвестный метод нормализации. Используйте 'minmax' или 'standard'.")
            
        print(f"Данные нормализованы методом: {method}")

    def train_test_split(self, test_size=0.2, shuffle=True):
        """
        Разбивает данные на обучающую и тестовую выборки.
        :param test_size: доля тестовой выборки (от 0 до 1)
        :param shuffle: перемешивать ли данные перед разбиением
        """
        if self.X is None or self.y is None:
            raise MismatchedDataError("Данные не загружены.")

        n_samples = self.X.shape[0]
        
        if shuffle:
            indices = np.random.permutation(n_samples)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
        else:
            X_shuffled = self.X
            y_shuffled = self.y

        split_idx = int(n_samples * (1 - test_size))

        self.X_train = X_shuffled[:split_idx]
        self.y_train = y_shuffled[:split_idx]
        self.X_test = X_shuffled[split_idx:]
        self.y_test = y_shuffled[split_idx:]

        print(f"Данные разделены: Обучающая - {self.X_train.shape[0]}, Тестовая - {self.X_test.shape[0]}")



    def normalize_single(self, x):
        """Нормализует один вектор входных данных для предсказания."""
        if self.min_vals is None:
            return x
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1
        return (x - self.min_vals) / range_vals