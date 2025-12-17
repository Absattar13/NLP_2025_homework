import pandas as pd
from dataset import Dataset
from model import Model

def main():
    # 1. Загружаем датасет
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    X = df['message'].tolist()
    y = df['label'].tolist()

    # 2. Создаём объект Dataset и делим на train/val/test
    dataset = Dataset(X, y)
    dataset.split_dataset(val=0.1, test=0.1)

    # 3. Обучаем модель
    model = Model(alpha=1)
    model.fit(dataset)

    # 4. Проверяем точность
    val_acc = model.validation()
    test_acc = model.test()
    print(f"📊 Validation Accuracy: {val_acc:.4f}")
    print(f"📊 Test Accuracy: {test_acc:.4f}")

    # 5. Пример предсказания
    sample = "Congratulations! You won a free ticket"
    result = model.inference(sample)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
