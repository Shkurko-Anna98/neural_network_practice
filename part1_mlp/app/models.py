"""
Модуль с реализацией полносвязной нейронной сети (многослойного перцептрона).
"""
import numpy as np
import json
from .exceptions import InvalidLayerSizeError, MismatchedDataError


class NeuralNetwork:
    """
    Класс полносвязной нейронной сети с одним или несколькими скрытыми слоями.
    """

    def __init__(self, layer_sizes, activation='sigmoid', random_seed=None):
        """
        Инициализация нейронной сети.
        :param layer_sizes: список размеров слоев, например [вход, скрытый, выход]
        :param activation: функция активации ('sigmoid' или 'relu')
        :param random_seed: зерно для генератора случайных чисел
        """
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError("Сеть должна иметь как минимум входной и выходной слои.")
        if any(size <= 0 for size in layer_sizes):
            raise InvalidLayerSizeError("Размеры всех слоев должны быть положительными числами.")

        self.layer_sizes = layer_sizes
        self.activation_name = activation.lower()
        self.history = []  # История ошибок для графика

        if random_seed is not None:
            np.random.seed(random_seed)

        # Инициализация весов и смещений
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # Заполняем веса случайными числами (нормальное распределение)
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        print("Нейронная сеть успешно создана!")

    def _activate(self, x):
        """Применяет функцию активации."""
        if self.activation_name == 'sigmoid':
            # clip нужен, чтобы избежать ошибки переполнения при больших числах
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.activation_name == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Неизвестная функция активации: {self.activation_name}")

    def _activate_derivative(self, x):
        """Производная функции активации (нужна для обратного прохода)."""
        if self.activation_name == 'sigmoid':
            sig = self._activate(x)
            return sig * (1 - sig)
        elif self.activation_name == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError(f"Неизвестная функция активации: {self.activation_name}")

    def forward(self, X):
        """
        Прямое распространение сигнала.
        :param X: входные данные
        :return: выходные данные и промежуточные значения
        """
        if X.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Число признаков во входных данных ({X.shape[1]}) не совпадает "
                f"с размером входного слоя ({self.layer_sizes[0]})."
            )

        layer_outputs = [X]
        layer_linear_outputs = []

        current_input = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(current_input, w) + b
            layer_linear_outputs.append(z)

            # Применяем активацию
            current_input = self._activate(z)
            layer_outputs.append(current_input)

        return layer_outputs, layer_linear_outputs

    def backward(self, layer_outputs, layer_linear_outputs, y_true, learning_rate):
        """
        Обратное распространение ошибки (Backpropagation).
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        m = y_true.shape[0]  # размер батча

        # Ошибка на выходном слое
        error = (layer_outputs[-1] - y_true) * self._activate_derivative(layer_linear_outputs[-1])

        # Идем от выходного слоя к входному
        for i in range(len(self.weights) - 1, -1, -1):
            # Градиенты
            dW = np.dot(layer_outputs[i].T, error) / m
            db = np.sum(error, axis=0, keepdims=True) / m

            # Обновляем параметры (градиентный спуск)
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

            # Передаем ошибку на предыдущий слой
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self._activate_derivative(layer_linear_outputs[i - 1])

    def train(self, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.01, verbose=True):
        """
        Обучение нейронной сети на батчах.
        """
        self.history = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Перемешиваем данные перед каждой эпохой
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            # Идем по данным порциями (батчами)
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # 1. Прямой проход
                layer_outputs, layer_linear_outputs = self.forward(X_batch)

                # 2. Считаем ошибку (MSE)
                loss = np.mean((layer_outputs[-1] - y_batch) ** 2)
                epoch_loss += loss

                # 3. Обратный проход
                self.backward(layer_outputs, layer_linear_outputs, y_batch, learning_rate)

            avg_loss = epoch_loss / (n_samples / batch_size)
            self.history.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Эпоха {epoch + 1}/{epochs}, Ошибка (MSE): {avg_loss:.6f}")

    def predict(self, X):
        """Предсказание для новых данных."""
        layer_outputs, _ = self.forward(X)
        return layer_outputs[-1]

    def save_weights(self, filepath):
        """Сохранение весов в JSON."""
        data = {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation_name,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def load_weights(self, filepath):
        """Загрузка весов из JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.layer_sizes = data['layer_sizes']
        self.activation_name = data['activation']
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]