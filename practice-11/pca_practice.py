import matplotlib

matplotlib.use("Agg")

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


# --------------------------------------------------
# 1. Загрузка данных
# --------------------------------------------------

dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

print("=== Данные ===")
print("Объектов:", X.shape[0])
print("Исходных признаков:", X.shape[1])
print("Пропусков:", int(X.isna().sum().sum()))
print("Классы:", y.value_counts().sort_index().to_dict())
print("0 = malignant, 1 = benign")


# --------------------------------------------------
# 2. PCA для анализа объяснённой дисперсии
# Эта PCA используется только для анализа и графиков.
# Оценка моделей ниже выполняется отдельным пайплайном.
# --------------------------------------------------

scaler_visual = StandardScaler()
X_scaled = scaler_visual.fit_transform(X)

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

components_90 = int(
    np.argmax(cumulative >= 0.90) + 1
)

print("\n=== Объяснённая дисперсия ===")
print("Первые 10 компонент:")
print(np.round(explained[:10], 4))

print(
    "Компонент для сохранения 90% дисперсии:",
    components_90,
)

print(
    "Дисперсия первых двух компонент:",
    round(float(cumulative[1]), 4),
)


# --------------------------------------------------
# 3. График объяснённой дисперсии
# --------------------------------------------------

component_numbers = np.arange(
    1,
    len(explained) + 1,
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    component_numbers,
    explained,
    alpha=0.55,
    color="royalblue",
    label="Доля отдельной компоненты",
)

ax.plot(
    component_numbers,
    cumulative,
    marker="o",
    markersize=4,
    color="darkorange",
    label="Накопленная дисперсия",
)

ax.axhline(
    0.90,
    color="red",
    linestyle="--",
    label="90% дисперсии",
)

ax.axvline(
    components_90,
    color="green",
    linestyle=":",
    label=f"{components_90} компонент",
)

ax.set_xlabel("Количество компонент")
ax.set_ylabel("Доля объяснённой дисперсии")
ax.set_title("PCA: накопленная объяснённая дисперсия")
ax.set_xticks(component_numbers)
ax.grid(alpha=0.25)
ax.legend()

plt.tight_layout()
plt.savefig("explained_variance.png", dpi=160)
plt.close()


# --------------------------------------------------
# 4. Проекция на две компоненты
# --------------------------------------------------

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

for class_value, class_name in enumerate(
    dataset.target_names
):
    mask = y.to_numpy() == class_value

    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        label=class_name,
        alpha=0.7,
        s=30,
    )

plt.xlabel(
    f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})"
)
plt.ylabel(
    f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})"
)
plt.title("Проекция медицинских данных на две компоненты")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=160)
plt.close()


# --------------------------------------------------
# 5. Честное разбиение для сравнения моделей
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE,
)


def evaluate_model(name, model):
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - start

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    test_probability = model.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(
        y_train,
        train_prediction,
    )
    test_accuracy = accuracy_score(
        y_test,
        test_prediction,
    )
    test_auc = roc_auc_score(
        y_test,
        test_probability,
    )

    if "pca" in model.named_steps:
        n_features = int(
            model.named_steps["pca"].n_components_
        )
    else:
        n_features = X_train.shape[1]

    return {
        "model": name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_roc_auc": test_auc,
        "fit_time_sec": fit_time,
        "features": n_features,
    }


# --------------------------------------------------
# 6. Базовая модель без PCA
# --------------------------------------------------

models = {
    "Без PCA": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
    "PCA 80%": Pipeline([
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                n_components=0.80,
                svd_solver="full",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
    "PCA 90%": Pipeline([
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                n_components=0.90,
                svd_solver="full",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
    "PCA 95%": Pipeline([
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                n_components=0.95,
                svd_solver="full",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
}


# --------------------------------------------------
# 7. Обучение и сравнение
# --------------------------------------------------

results = []

for name, model in models.items():
    results.append(
        evaluate_model(name, model)
    )

results_df = pd.DataFrame(results)

print("\n=== Сравнение моделей ===")
print(
    results_df.to_string(
        index=False,
        formatters={
            "train_accuracy": "{:.4f}".format,
            "test_accuracy": "{:.4f}".format,
            "test_roc_auc": "{:.4f}".format,
            "fit_time_sec": "{:.4f}".format,
        },
    )
)

results_df.to_csv(
    "pca_results.csv",
    index=False,
)


# --------------------------------------------------
# 8. График сравнения
# --------------------------------------------------

fig, ax1 = plt.subplots(figsize=(10, 6))

positions = np.arange(len(results_df))

ax1.bar(
    positions - 0.18,
    results_df["test_accuracy"],
    width=0.36,
    label="Accuracy",
    color="royalblue",
)

ax1.bar(
    positions + 0.18,
    results_df["test_roc_auc"],
    width=0.36,
    label="ROC-AUC",
    color="darkorange",
)

ax1.set_ylim(0.85, 1.01)
ax1.set_ylabel("Качество")
ax1.set_xticks(positions)
ax1.set_xticklabels(
    results_df["model"],
    rotation=15,
)
ax1.grid(axis="y", alpha=0.25)
ax1.legend(loc="lower left")

for position, features in zip(
    positions,
    results_df["features"],
):
    ax1.text(
        position,
        0.855,
        f"{features} признаков",
        ha="center",
        fontsize=9,
    )

plt.title("Качество модели без PCA и с PCA")
plt.tight_layout()
plt.savefig("pca_model_comparison.png", dpi=160)
plt.close()


# --------------------------------------------------
# 9. Итог
# --------------------------------------------------

best_row = results_df.loc[
    results_df["test_roc_auc"].idxmax()
]

print("\n=== Вывод ===")
print(
    "Лучшая модель по ROC-AUC:",
    best_row["model"],
)
print(
    "ROC-AUC:",
    round(best_row["test_roc_auc"], 4),
)
print(
    "Число признаков:",
    int(best_row["features"]),
)

print("\n=== Созданные файлы ===")
print("explained_variance.png")
print("pca_2d.png")
print("pca_model_comparison.png")
print("pca_results.csv")
