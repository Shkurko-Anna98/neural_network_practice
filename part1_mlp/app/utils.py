"""
Вспомогательные функции для отображения результатов.
"""
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Строит график зависимости ошибки от эпохи обучения.
    :param history: список значений ошибки по эпохам
    """
    if not history:
        print("История обучения пуста. Сначала обучите сеть.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='b')
    plt.title('График ошибки (MSE) по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка (MSE)')
    plt.grid(True)
    plt.show()