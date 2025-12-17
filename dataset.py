import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X  # сообщения
        self._y = y  # метки ["spam", "ham"]
        self.train = None  # кортеж из (X_train, y_train)
        self.val = None    # кортеж из (X_val, y_val)
        self.test = None   # кортеж из (X_test, y_test)
        self.label2num = {}  # словарь: метка -> число
        self.num2label = {}  # словарь: число -> метка
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _clean_text(self, text: str) -> str:
        """
        Очистка текста: удаляем лишние символы, приводим к нижнему регистру.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # оставляем только буквы/цифры/пробелы
        text = re.sub(r"\s+", " ", text).strip()  # убираем лишние пробелы
        return text
    
    def _transform(self):
        """
        Функция очистки сообщений и преобразования меток в числа.
        """
        # Очистка сообщений
        self._x = [self._clean_text(msg) for msg in self._x]

        # Создание словарей для меток
        unique_labels = sorted(set(self._y))
        self.label2num = {label: idx for idx, label in enumerate(unique_labels)}
        self.num2label = {idx: label for label, idx in self.label2num.items()}

        # Преобразование меток в числа
        self._y = np.array([self.label2num[label] for label in self._y])
    
    def split_dataset(self, val=0.1, test=0.1):
        """
        Разбивает набор данных на train/val/test.
        """
        n = len(self._x)
        indices = np.arange(n)
        np.random.shuffle(indices)

        test_size = int(n * test)
        val_size = int(n * val)
        train_size = n - test_size - val_size

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]

        X = np.array(self._x)
        y = np.array(self._y)

        self.train = (X[train_idx], y[train_idx])
        self.val = (X[val_idx], y[val_idx])
        self.test = (X[test_idx], y[test_idx])
