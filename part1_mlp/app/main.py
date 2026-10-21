"""
Главный модуль приложения. Консольное меню для работы с нейросетью.
"""
import numpy as np
from .models import NeuralNetwork
from .manager import DatasetManager
from .exceptions import NeuralNetworkError, MismatchedDataError
from .utils import plot_training_history


def main_menu():
    net = None
    dm = DatasetManager()

    while True:
        print("\n" + "="*40)
        print("   МЕНЮ НЕЙРОСЕТИ (Часть 1)")
        print("="*40)
        print("1. Создать сеть")
        print("2. Загрузить данные из CSV")
        print("3. Обучить сеть")
        print("4. Сделать предсказание")
        print("5. Сохранить/Загрузить веса")
        print("6. Показать график ошибки")
        print("7. Выход")
        print("="*40)
        
        choice = input("Выберите пункт меню (1-7): ").strip()
        
        if choice == '1':
            try:
                print("\n--- Создание сети ---")
                layers_str = input("Введите размеры слоев через запятую (например, 4,8,1): ")
                layers = [int(x.strip()) for x in layers_str.split(',') if x.strip()]
                
                act = input("Выберите функцию активации (sigmoid / relu): ").strip().lower()
                if act not in ['sigmoid', 'relu']:
                    print("Ошибка: Доступны 'sigmoid' или 'relu'. Выбрана 'sigmoid' по умолчанию.")
                    act = 'sigmoid'
                    
                net = NeuralNetwork(layer_sizes=layers, activation=act)
            except ValueError:
                print("Ошибка: Введите целые числа через запятую.")
            except NeuralNetworkError as e:
                print(f"Ошибка нейросети: {e}")
                
        elif choice == '2':
            try:
                print("\n--- Загрузка данных ---")
                path = input("Введите путь к CSV-файлу: ").strip('"')
                dm.load_csv(path)
                
                norm_choice = input("Нормализовать данные? (1 - MinMax, 2 - Standard, 0 - Пропустить): ").strip()
                if norm_choice == '1':
                    dm.normalize(method='minmax')
                elif norm_choice == '2':
                    dm.normalize(method='standard')
                
                dm.train_test_split(test_size=0.2)
            except NeuralNetworkError as e:
                print(f"Ошибка: {e}")
                
        elif choice == '3':
            if net is None:
                print("Сначала создайте сеть (пункт 1)!")
                continue
            if dm.X_train is None:
                print("Сначала загрузите и подготовьте данные (пункт 2)!")
                continue
                
            try:
                print("\n--- Обучение сети ---")
                lr = float(input("Введите скорость обучения (например, 0.1): "))
                epochs = int(input("Введите количество эпох (например, 100): "))
                batch_size = int(input("Введите размер батча (например, 32): "))
                
                # Проверка на соответствие размеров
                if dm.X_train.shape[1] != net.layer_sizes[0]:
                     raise MismatchedDataError("Размер входного слоя сети не совпадает с числом признаков в данных!")
                
                net.train(dm.X_train, dm.y_train, epochs=epochs, batch_size=batch_size, learning_rate=lr)
                
                # Оценим на тесте
                preds = net.predict(dm.X_test)
                mse = np.mean((preds.flatten() - dm.y_test) ** 2)
                print(f"Ошибка (MSE) на тестовой выборке: {mse:.6f}")
                
            except ValueError:
                print("Ошибка ввода: проверьте числа.")
            except NeuralNetworkError as e:
                print(f"Ошибка: {e}")
                
        elif choice == '4':
            if net is None:
                print("Сначала создайте или загрузите сеть!")
                continue
            try:
                print("\n--- Предсказание ---")
                print("Введите признаки через запятую (в том же порядке, что и в CSV):")
                inp = input("> ")
                x_vals = [float(x.strip()) for x in inp.split(',') if x.strip()]
                
                if len(x_vals) != net.layer_sizes[0]:
                     raise MismatchedDataError(f"Ожидалось {net.layer_sizes[0]} признаков, получено {len(x_vals)}")
                
                x_arr = np.array(x_vals).reshape(1, -1)
                # Применяем ту же нормализацию, что и при обучении
                x_arr = dm.normalize_single(x_arr)
                
                pred = net.predict(x_arr)
                print(f"Результат предсказания: {pred[0][0]:.4f}")
                
            except ValueError:
                 print("Ошибка: Введите числа через запятую.")
            except NeuralNetworkError as e:
                print(f"Ошибка: {e}")
                
        elif choice == '5':
            if net is None:
                print("Сначала создайте сеть!")
                continue
            print("\n--- Работа с весами ---")
            action = input("1 - Сохранить, 2 - Загрузить: ").strip()
            if action == '1':
                path = input("Введите имя файла для сохранения (например, weights.json): ").strip()
                net.save_weights(path)
                print("Веса сохранены.")
            elif action == '2':
                path = input("Введите имя файла для загрузки: ").strip()
                try:
                    net.load_weights(path)
                    print("Веса загружены.")
                except Exception as e:
                    print(f"Ошибка загрузки: {e}")
            else:
                print("Неверный выбор.")
                
        elif choice == '6':
            plot_training_history(net.history if net else [])
            
        elif choice == '7':
            print("Выход из программы. До свидания!")
            break
            
        else:
            print("Неверный пункт меню. Попробуйте снова.")

if __name__ == "__main__":
    main_menu()