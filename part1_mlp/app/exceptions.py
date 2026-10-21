"""
Модуль с пользовательскими исключениями для нейросети.
"""

class NeuralNetworkError(Exception):
    """Базовый класс для всех ошибок нашей нейросети."""
    pass

class InvalidLayerSizeError(NeuralNetworkError):
    """Ошибка, возникающая при неверном размере слоев (например, отрицательном или нулевом)."""
    pass

class MismatchedDataError(NeuralNetworkError):
    """Ошибка несовместимости размерностей данных (например, если число признаков не совпадает с входным слоем)."""
    pass

class FileLoadError(NeuralNetworkError):
    """Ошибка при загрузке файла (например, файл не найден или поврежден)."""
    pass

class InvalidInputError(NeuralNetworkError):
    """Ошибка при вводе некорректных данных пользователем."""
    pass