"""
Функции для аугментации изображений (повороты, отражения, шум).
"""
import numpy as np
from PIL import Image

def apply_augmentation(img_array):
    """
    Принимает картинку как массив, делает случайное преобразование и возвращает обратно.
    """
    # Переводим массив обратно в картинку PIL
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # Случайно выбираем действие (0, 1 или 2)
    action = np.random.randint(0, 3)
    
    if action == 0:
        # Поворот на 90 градусов
        img = img.rotate(90)
    elif action == 1:
        # Отражение слева направо
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        # Добавление шума
        noise = np.random.normal(0, 10, img_array.shape).astype(np.float32)
        noisy_arr = img_array + noise
        # Обрезаем значения, чтобы не выходили за пределы [0, 1]
        noisy_arr = np.clip(noisy_arr, 0.0, 1.0)
        return noisy_arr

    # Если было поворот или отражение, переводим обратно в массив
    res_arr = np.array(img, dtype=np.float32) / 255.0
    return res_arr