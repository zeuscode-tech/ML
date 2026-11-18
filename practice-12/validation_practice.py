from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import loguniform

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "o_churn_synthetic.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_model():
    """Полный пайплайн: предобработка и логистическая регрессия."""
    categorical_columns = ["region", "tariff"]

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            ),
        ],
    )

    model = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def single_split_score(random_state):
    """Оценка модели на одном разбиении development-выборки."""
    x_train, x_valid, y_train, y_valid = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=random_state,
    )

    pipeline = make_model()

    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    elapsed = time.perf_counter() - started

    probability = pipeline.predict_proba(x_valid)[:, 1]
    score = roc_auc_score(y_valid, probability)

    return score, elapsed


def evaluate_on_test(name, model):
    """Финальная оценка на отложенной тестовой выборке."""
    started = time.perf_counter()
    model.fit(X_development, y_development)
    elapsed = time.perf_counter() - started

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy_test": accuracy_score(y_test, prediction),
        "f1_test": f1_score(y_test, prediction),
        "roc_auc_test": roc_auc_score(y_test, probability),
        "fit_time_sec": elapsed,
    }


# ============================================================
# Загрузка данных
# ============================================================

data = pd.read_csv(DATA_PATH)

X = data.drop(
    columns=[
        "churn",
        "subscriber_code",
    ]
)

y = data["churn"]

print("=== Данные ===")
print("Строк:", len(data))
print("Исходных признаков:", X.shape[1])
print("Пропусков:", int(data.isna().sum().sum()))
print("Распределение классов:")
print(y.value_counts().sort_index())
print("Доля churn=1:", round(float(y.mean()), 4))


# ============================================================
# Отдельная тестовая выборка
# ============================================================

X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("\n=== Честное разбиение ===")
print("Development:", X_development.shape)
print("Test:", X_test.shape)
print(
    "Test не участвует в кросс-валидации "
    "и подборе гиперпараметров."
)


# ============================================================
# Уровень A: одиночное разбиение
# ============================================================

score_state_42, time_state_42 = single_split_score(42)
score_state_7, time_state_7 = single_split_score(7)

print("\n=== Одиночные разбиения ===")
print(
    f"random_state=42: ROC-AUC={score_state_42:.4f}, "
    f"время={time_state_42:.4f} с"
)
print(
    f"random_state=7:  ROC-AUC={score_state_7:.4f}, "
    f"время={time_state_7:.4f} с"
)
print(
    "Разница между разбиениями:",
    round(abs(score_state_42 - score_state_7), 4),
)


# ============================================================
# Уровень A: 5-кратная кросс-валидация
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

baseline_pipeline = make_model()

cv_started = time.perf_counter()

cv_scores = cross_val_score(
    baseline_pipeline,
    X_development,
    y_development,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
)

cv_time = time.perf_counter() - cv_started

print("\n=== 5-кратная кросс-валидация ===")

for number, score in enumerate(cv_scores, start=1):
    print(f"Фолд {number}: ROC-AUC={score:.4f}")

print("Среднее:", round(float(cv_scores.mean()), 4))
print("Стандартное отклонение:", round(float(cv_scores.std()), 4))
print(
    "Размах:",
    round(float(cv_scores.max() - cv_scores.min()), 4),
)
print("Время CV:", round(cv_time, 4), "с")

cv_table = pd.DataFrame(
    {
        "fold": np.arange(1, 6),
        "roc_auc": cv_scores,
    }
)

cv_table.to_csv(
    HERE / "cv_scores.csv",
    index=False,
)


# ============================================================
# График устойчивости оценки
# ============================================================

labels = [
    "Split 42",
    "Split 7",
    "Fold 1",
    "Fold 2",
    "Fold 3",
    "Fold 4",
    "Fold 5",
]

values = [
    score_state_42,
    score_state_7,
    *cv_scores,
]

plt.figure(figsize=(10, 5))

colors = [
    "#E76F51",
    "#F4A261",
    "#2A9D8F",
    "#2A9D8F",
    "#2A9D8F",
    "#2A9D8F",
    "#2A9D8F",
]

bars = plt.bar(
    labels,
    values,
    color=colors,
)

plt.axhline(
    cv_scores.mean(),
    color="black",
    linestyle="--",
    label=f"Среднее CV = {cv_scores.mean():.3f}",
)

plt.ylim(
    max(0.5, min(values) - 0.04),
    min(1.0, max(values) + 0.04),
)

plt.ylabel("ROC-AUC")
plt.title("Одиночные разбиения и кросс-валидация")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.25)
plt.legend()

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.003,
        f"{value:.3f}",
        ha="center",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(
    HERE / "validation_stability.png",
    dpi=160,
)
plt.close()


# ============================================================
# Уровень B: GridSearchCV
# ============================================================

grid_parameters = {
    "model__C": [
        0.01,
        0.1,
        1.0,
        10.0,
    ],
    "model__class_weight": [
        None,
        "balanced",
    ],
}

grid_search = GridSearchCV(
    estimator=make_model(),
    param_grid=grid_parameters,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

grid_started = time.perf_counter()
grid_search.fit(X_development, y_development)
grid_time = time.perf_counter() - grid_started

print("\n=== GridSearchCV ===")
print("Проверено комбинаций:", len(grid_search.cv_results_["params"]))
print("Лучшие параметры:", grid_search.best_params_)
print("Лучший ROC-AUC CV:", round(grid_search.best_score_, 4))
print("Время:", round(grid_time, 4), "с")


# ============================================================
# Уровень B: RandomizedSearchCV
# ============================================================

random_parameters = {
    "model__C": loguniform(0.001, 100),
    "model__class_weight": [
        None,
        "balanced",
    ],
}

random_search = RandomizedSearchCV(
    estimator=make_model(),
    param_distributions=random_parameters,
    n_iter=8,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    refit=True,
    return_train_score=True,
)

random_started = time.perf_counter()
random_search.fit(X_development, y_development)
random_time = time.perf_counter() - random_started

print("\n=== RandomizedSearchCV ===")
print("Проверено комбинаций:", len(random_search.cv_results_["params"]))
print("Лучшие параметры:", random_search.best_params_)
print("Лучший ROC-AUC CV:", round(random_search.best_score_, 4))
print("Время:", round(random_time, 4), "с")


# ============================================================
# Выбор победителя только по CV, а не по test
# ============================================================

if grid_search.best_score_ >= random_search.best_score_:
    winner_name = "GridSearchCV"
    winner_search = grid_search
else:
    winner_name = "RandomizedSearchCV"
    winner_search = random_search

print("\n=== Победитель поиска ===")
print("Метод:", winner_name)
print("Параметры:", winner_search.best_params_)
print("ROC-AUC CV:", round(winner_search.best_score_, 4))


# ============================================================
# Итоговая оценка на test
# ============================================================

baseline_result = evaluate_on_test(
    "Базовая модель",
    make_model(),
)

tuned_result = evaluate_on_test(
    f"Настроенная модель ({winner_name})",
    winner_search.best_estimator_,
)

final_results = pd.DataFrame(
    [
        baseline_result,
        tuned_result,
    ]
)

final_results.to_csv(
    HERE / "final_test_metrics.csv",
    index=False,
)

print("\n=== Финальная оценка на test ===")
print(
    final_results.to_string(
        index=False,
        formatters={
            "accuracy_test": "{:.4f}".format,
            "f1_test": "{:.4f}".format,
            "roc_auc_test": "{:.4f}".format,
            "fit_time_sec": "{:.4f}".format,
        },
    )
)


# ============================================================
# Сравнение Grid Search и Random Search
# ============================================================

search_comparison = pd.DataFrame(
    [
        {
            "method": "GridSearchCV",
            "iterations": len(grid_search.cv_results_["params"]),
            "best_cv_roc_auc": grid_search.best_score_,
            "search_time_sec": grid_time,
            "best_parameters": str(grid_search.best_params_),
        },
        {
            "method": "RandomizedSearchCV",
            "iterations": len(random_search.cv_results_["params"]),
            "best_cv_roc_auc": random_search.best_score_,
            "search_time_sec": random_time,
            "best_parameters": str(random_search.best_params_),
        },
    ]
)

search_comparison.to_csv(
    HERE / "search_comparison.csv",
    index=False,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
)

axes[0].bar(
    search_comparison["method"],
    search_comparison["best_cv_roc_auc"],
    color=["#457B9D", "#E76F51"],
)

axes[0].set_title("Лучший ROC-AUC на CV")
axes[0].set_ylabel("ROC-AUC")
axes[0].set_ylim(
    max(
        0.5,
        search_comparison["best_cv_roc_auc"].min() - 0.03,
    ),
    min(
        1.0,
        search_comparison["best_cv_roc_auc"].max() + 0.03,
    ),
)
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(
    search_comparison["method"],
    search_comparison["search_time_sec"],
    color=["#457B9D", "#E76F51"],
)

axes[1].set_title("Время поиска")
axes[1].set_ylabel("Секунды")
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    HERE / "search_comparison.png",
    dpi=160,
)
plt.close()


# ============================================================
# Общий вывод
# ============================================================

print("\n=== Сравнение трёх оценок ===")
print(
    "Базовая модель, средний CV:",
    round(float(cv_scores.mean()), 4),
)
print(
    "Лучшая оценка после поиска:",
    round(float(winner_search.best_score_), 4),
)
print(
    "Настроенная модель на test:",
    round(float(tuned_result["roc_auc_test"]), 4),
)

bias = winner_search.best_score_ - tuned_result["roc_auc_test"]

print(
    "Разница best CV и test:",
    round(float(bias), 4),
)

print("\n=== Вывод ===")

if cv_scores.std() > 0.05:
    print(
        "Разброс по фолдам превышает 5 процентных пунктов. "
        "Оценка по одному разбиению ненадёжна."
    )
else:
    print(
        "Разброс по фолдам умеренный, но кросс-валидация "
        "всё равно надёжнее одного разбиения."
    )

print(
    "Grid Search проверяет заданную сетку полностью, "
    "а Random Search исследует случайные комбинации."
)

print(
    "Лучшая CV-оценка может быть оптимистичной, потому что "
    "она выбрана как максимум среди нескольких вариантов."
)

print(
    "Тестовая выборка не участвовала в подборе параметров "
    "и использована только для итогового отчёта."
)

print("\n=== Созданные файлы ===")
print("cv_scores.csv")
print("validation_stability.png")
print("search_comparison.csv")
print("search_comparison.png")
print("final_test_metrics.csv")


def main():
    """Вычисления выполняются при запуске файла."""
    pass


if __name__ == "__main__":
    main()
