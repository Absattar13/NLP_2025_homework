import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Деректерді жүктеу
    df = pd.read_csv("t_zz_text.csv", sep="|", on_bad_lines='skip', low_memory=False)
    df = df[['transcript_operator_words', 'name_1']].dropna()

    # 2. Сирек категорияларды алып тастау
    counts = df['name_1'].value_counts()
    df = df[df['name_1'].isin(counts[counts > 30].index)]

    # 3. Входные/выходные
    X = df['transcript_operator_words']
    y = df['name_1']

    # 4. Train/Test бөлу
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. TF-IDF векторизация
    vectorizer = TfidfVectorizer(stop_words=None)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 6. Наивный Байес
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # 7. Метрикалар (Classification Report)
    y_pred = model.predict(X_test_tfidf)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    classes = list(set(y_test))  

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # 9. Пример предсказания
    sample = "Здравствуйте, хочу узнать условия по кредитной карте"
    result = model.predict(vectorizer.transform([sample]))[0]
    print(f"\nSample: '{sample}' → {result}")

if __name__ == "__main__":
    main()
