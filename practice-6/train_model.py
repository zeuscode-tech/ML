#!/usr/bin/env python3

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/SMSSpamCollection")
RANDOM_STATE = 42

# Отрицания нельзя удалять: они могут менять смысл текста.
NEGATIONS = {"no", "not", "never", "nor"}

STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS


def clean_text(text):
    """
    Очистка одного текста:
    1. нижний регистр;
    2. токенизация;
    3. удаление цифр и пунктуации;
    4. удаление стоп-слов.
    """
    text = str(text).lower()
    # Раскрываем отрицания до удаления пунктуации.
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)

    # Регулярное выражение оставляет только английские буквы.
    tokens = re.findall(r"[a-z]+", text)

    cleaned_tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    return " ".join(cleaned_tokens)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Собственный этап очистки для sklearn Pipeline."""

    def fit(self, texts, labels=None):
        return self

    def transform(self, texts):
        return [
            clean_text(text)
            for text in texts
        ]


def explain_error(true_label, predicted_label, text):
    """Формирует предварительную гипотезу о причине ошибки."""
    lowered = text.lower()

    promotional_words = {
        "free",
        "win",
        "winner",
        "prize",
        "claim",
        "cash",
        "offer",
        "urgent",
        "call",
        "text",
    }

    found_promotional = sorted(
        word
        for word in promotional_words
        if word in lowered
    )

    if true_label == "spam" and predicted_label == "ham":
        if len(text.split()) < 8:
            return (
                "Короткий спам: недостаточно характерных "
                "слов для уверенного решения."
            )

        return (
            "Спам похож на обычное личное сообщение или "
            "содержит редкие слова, которых мало в обучении."
        )

    if true_label == "ham" and predicted_label == "spam":
        if found_promotional:
            return (
                "Обычное сообщение содержит слова, часто "
                "встречающиеся в спаме: "
                + ", ".join(found_promotional)
                + "."
            )

        return (
            "Обычное сообщение по лексике и структуре "
            "похоже на рекламное."
        )

    return "Причина требует дополнительного анализа."


def main():
    data = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["label", "text"],
    )

    print("=== Данные ===")
    print(f"Количество сообщений: {len(data)}")
    print(data["label"].value_counts())
    print()

    print("=== Примеры предобработки ===")

    for index in range(5):
        original = data.loc[index, "text"]
        cleaned = clean_text(original)

        print(f"\nПример {index + 1}")
        print(f"До:    {original}")
        print(f"После: {cleaned}")

    # Сначала выполняется разбиение.
    # Векторизатор ещё не видел тестовую выборку.
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    pipeline = Pipeline(
        steps=[
            ("cleaner", TextCleaner()),
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=2,
                    max_df=0.95,
                    max_features=5000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                MultinomialNB(alpha=0.5),
            ),
        ]
    )

    print("\n=== Обучение ===")
    print(f"Train: {len(x_train)}")
    print(f"Test:  {len(x_test)}")

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)

    tfidf = pipeline.named_steps["tfidf"]
    vocabulary_size = len(tfidf.vocabulary_)

    cleaned_train = pipeline.named_steps[
        "cleaner"
    ].transform(x_train)

    train_matrix = tfidf.transform(cleaned_train)

    print(
        "Размерность TF-IDF train: "
        f"{train_matrix.shape}"
    )
    print(
        f"Размер словаря: {vocabulary_size}"
    )

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        pos_label="spam",
    )
    recall = recall_score(
        y_test,
        predictions,
        pos_label="spam",
    )
    f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam",
    )

    print("\n=== Метрики на тестовой выборке ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\n=== Classification report ===")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"],
    )

    print("=== Матрица ошибок ===")
    print(matrix)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок: SMS Spam")
    plt.tight_layout()
    plt.savefig(
        "confusion_matrix.png",
        dpi=160,
    )
    plt.close()

    class_indices = {
        label: index
        for index, label in enumerate(
            pipeline.named_steps[
                "classifier"
            ].classes_
        )
    }

    error_rows = []

    for position, (
        original_index,
        text,
        true_label,
        predicted_label,
    ) in enumerate(
        zip(
            x_test.index,
            x_test,
            y_test,
            predictions,
        )
    ):
        if true_label == predicted_label:
            continue

        confidence = probabilities[
            position,
            class_indices[predicted_label],
        ]

        error_rows.append(
            {
                "original_index": original_index,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "text": text,
                "cleaned_text": clean_text(text),
                "hypothesis": explain_error(
                    true_label,
                    predicted_label,
                    text,
                ),
            }
        )

    errors = pd.DataFrame(error_rows)
    errors = errors.sort_values(
        "confidence",
        ascending=False,
    )

    errors.to_csv(
        "all_errors.csv",
        index=False,
    )

    top_errors = errors.head(5).copy()
    top_errors.to_csv(
        "top5_errors.csv",
        index=False,
    )

    print("\n=== Пять ошибок модели ===")

    for number, (_, row) in enumerate(
        top_errors.iterrows(),
        1,
    ):
        print(f"\nОшибка {number}")
        print(
            f"Истинный класс: "
            f"{row['true_label']}"
        )
        print(
            f"Предсказание: "
            f"{row['predicted_label']}"
        )
        print(
            f"Уверенность: "
            f"{row['confidence']:.4f}"
        )
        print(f"Текст: {row['text']}")
        print(
            f"После очистки: "
            f"{row['cleaned_text']}"
        )
        print(
            f"Гипотеза: "
            f"{row['hypothesis']}"
        )

    print("\n=== Созданные файлы ===")
    print("confusion_matrix.png")
    print("all_errors.csv")
    print("top5_errors.csv")


if __name__ == "__main__":
    main()
