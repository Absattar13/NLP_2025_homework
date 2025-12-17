import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set()   # словарь всех уникальных слов из train
        self.spam = {}       # частоты слов в спам-сообщениях
        self.ham = {}        # частоты слов в ham-сообщениях
        self.alpha = alpha   # сглаживание
        self.label2num = None
        self.num2label = None
        self.Nvoc = None     # количество уникальных слов
        self.Nspam = None    # количество слов в спам-сообщениях
        self.Nham = None     # количество слов в ham-сообщениях
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def _tokenize(self, text):
        """
        Разбиваем сообщение на слова, оставляем только буквы/цифры.
        """
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    def fit(self, dataset):
        """
        dataset - объект класса Dataset
        Заполняем атрибуты модели на основе train/val/test из dataset.
        """
        # сохраняем train/val/test
        self._train_X, self._train_y = dataset.train
        self._val_X, self._val_y = dataset.val
        self._test_X, self._test_y = dataset.test

        self.label2num = dataset.label2num
        self.num2label = dataset.num2label

        # считаем частоты слов
        spam_counts = {}
        ham_counts = {}
        vocab = set()

        for msg, label in zip(self._train_X, self._train_y):
            tokens = self._tokenize(msg)
            for token in tokens:
                vocab.add(token)
                if label == self.label2num["spam"]:
                    spam_counts[token] = spam_counts.get(token, 0) + 1
                else:
                    ham_counts[token] = ham_counts.get(token, 0) + 1

        self.vocab = vocab
        self.spam = spam_counts
        self.ham = ham_counts
        self.Nvoc = len(vocab)
        self.Nspam = sum(spam_counts.values())
        self.Nham = sum(ham_counts.values())

    def inference(self, message):
        """
        Наивный байес: считаем вероятность спама и хама для сообщения.
        """
        tokens = self._tokenize(message)

        # априорные вероятности
        prior_spam = sum(self._train_y == self.label2num["spam"]) / len(self._train_y)
        prior_ham = sum(self._train_y == self.label2num["ham"]) / len(self._train_y)

        pspam = np.log(prior_spam)
        pham = np.log(prior_ham)

        for token in tokens:
            # вероятность слова при спаме
            p_w_spam = (self.spam.get(token, 0) + self.alpha) / (self.Nspam + self.alpha * self.Nvoc)
            # вероятность слова при ham
            p_w_ham = (self.ham.get(token, 0) + self.alpha) / (self.Nham + self.alpha * self.Nvoc)

            pspam += np.log(p_w_spam)
            pham += np.log(p_w_ham)

        if pspam > pham:
            return "spam"
        return "ham"

    def validation(self):
        """
        Предсказываем метки для validation и считаем точность.
        """
        preds = [self.inference(msg) for msg in self._val_X]
        true_labels = [self.num2label[y] for y in self._val_y]
        correct = sum(p == t for p, t in zip(preds, true_labels))
        val_acc = correct / len(true_labels)
        return val_acc

    def test(self):
        """
        Предсказываем метки для test и считаем точность.
        """
        preds = [self.inference(msg) for msg in self._test_X]
        true_labels = [self.num2label[y] for y in self._test_y]
        correct = sum(p == t for p, t in zip(preds, true_labels))
        test_acc = correct / len(true_labels)
        return test_acc
