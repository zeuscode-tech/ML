from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from scipy.stats import loguniform

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_model(
    c_value=1.0,
    penalty="l2",
    class_weight="balanced",
):
    """Полный пайплайн без утечки данных."""
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
        ]
    )

    classifier = LogisticRegression(
        C=c_value,
        penalty=penalty,
        solver="liblinear",
        class_weight=class_weight,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


# ============================================================
# Данные
# ============================================================

data = pd.read_csv(DATA_PATH)

X = data.drop(
    columns=[
        "churn",
        "subscriber_code",
    ]
)

y = data["churn"]

X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("=== Данные ===")
print("Всего объектов:", len(data))
print("Development:", X_development.shape)
print("Отложенный test:", X_test.shape)
print("Доля churn=1:", round(float(y.mean()), 4))
print("Test не используется во вложенной CV.")


# ============================================================
# Сетка параметров
# ============================================================

parameter_grid = {
    "model__C": [
        0.01,
        0.1,
        1.0,
        10.0,
    ],
    "model__penalty": [
        "l1",
        "l2",
    ],
}

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

ordinary_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# Обычный GridSearchCV
# ============================================================

ordinary_search = GridSearchCV(
    estimator=make_model(),
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=ordinary_cv,
    n_jobs=-1,
    refit=True,
)

ordinary_started = time.perf_counter()
ordinary_search.fit(X_development, y_development)
ordinary_time = time.perf_counter() - ordinary_started

print("\n=== Обычный GridSearchCV ===")
print("Комбинаций:", len(ordinary_search.cv_results_["params"]))
print("Лучшие параметры:", ordinary_search.best_params_)
print(
    "Лучшая оценка CV:",
    round(float(ordinary_search.best_score_), 4),
)
print("Время:", round(ordinary_time, 3), "с")


# ============================================================
# Вложенная кросс-валидация
# ============================================================

nested_rows = []

nested_started = time.perf_counter()

for fold_number, (train_indices, valid_indices) in enumerate(
    outer_cv.split(X_development, y_development),
    start=1,
):
    x_train = X_development.iloc[train_indices]
    x_valid = X_development.iloc[valid_indices]

    y_train = y_development.iloc[train_indices]
    y_valid = y_development.iloc[valid_indices]

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE + fold_number,
    )

    inner_search = GridSearchCV(
        estimator=make_model(),
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )

    inner_search.fit(x_train, y_train)

    probability = inner_search.predict_proba(x_valid)[:, 1]
    outer_score = roc_auc_score(y_valid, probability)

    row = {
        "outer_fold": fold_number,
        "outer_roc_auc": outer_score,
        "inner_best_score": inner_search.best_score_,
        "best_C": inner_search.best_params_["model__C"],
        "best_penalty": inner_search.best_params_[
            "model__penalty"
        ],
    }

    nested_rows.append(row)

    print(
        f"Внешний фолд {fold_number}: "
        f"ROC-AUC={outer_score:.4f}, "
        f"C={row['best_C']}, "
        f"penalty={row['best_penalty']}"
    )

nested_time = time.perf_counter() - nested_started

nested_results = pd.DataFrame(nested_rows)

nested_results.to_csv(
    HERE / "nested_cv_results.csv",
    index=False,
)

nested_mean = nested_results["outer_roc_auc"].mean()
nested_std = nested_results["outer_roc_auc"].std(ddof=0)

selection_bias = ordinary_search.best_score_ - nested_mean

print("\n=== Результат вложенной CV ===")
print("Средний ROC-AUC:", round(float(nested_mean), 4))
print("Стандартное отклонение:", round(float(nested_std), 4))
print("Время:", round(nested_time, 3), "с")
print(
    "Смещение обычной схемы:",
    round(float(selection_bias), 4),
)


# ============================================================
# График обычной и вложенной CV
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    nested_results["outer_fold"].astype(str),
    nested_results["outer_roc_auc"],
    color="#2A9D8F",
    label="Внешние фолды",
)

plt.axhline(
    nested_mean,
    color="#264653",
    linestyle="--",
    linewidth=2,
    label=f"Nested mean = {nested_mean:.3f}",
)

plt.axhline(
    ordinary_search.best_score_,
    color="#E76F51",
    linestyle=":",
    linewidth=2,
    label=(
        "Обычный GridSearchCV = "
        f"{ordinary_search.best_score_:.3f}"
    ),
)

plt.xlabel("Внешний фолд")
plt.ylabel("ROC-AUC")
plt.title("Обычная и вложенная кросс-валидация")
plt.ylim(
    max(
        0.5,
        nested_results["outer_roc_auc"].min() - 0.04,
    ),
    min(
        1.0,
        max(
            nested_results["outer_roc_auc"].max(),
            ordinary_search.best_score_,
        ) + 0.04,
    ),
)
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    HERE / "nested_cv_comparison.png",
    dpi=160,
)
plt.close()


# ============================================================
# Grid Search
# ============================================================

search_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

grid_search = GridSearchCV(
    estimator=make_model(),
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=search_cv,
    n_jobs=-1,
    refit=True,
)

grid_started = time.perf_counter()
grid_search.fit(X_development, y_development)
grid_time = time.perf_counter() - grid_started


# ============================================================
# Random Search
# ============================================================

random_parameters = {
    "model__C": loguniform(0.001, 100),
    "model__penalty": [
        "l1",
        "l2",
    ],
}

random_search = RandomizedSearchCV(
    estimator=make_model(),
    param_distributions=random_parameters,
    n_iter=8,
    scoring="roc_auc",
    cv=search_cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    refit=True,
)

random_started = time.perf_counter()
random_search.fit(X_development, y_development)
random_time = time.perf_counter() - random_started


# ============================================================
# Байесовская оптимизация Optuna
# ============================================================

def optuna_objective(trial):
    c_value = trial.suggest_float(
        "C",
        0.001,
        100.0,
        log=True,
    )

    penalty = trial.suggest_categorical(
        "penalty",
        [
            "l1",
            "l2",
        ],
    )

    model = make_model(
        c_value=c_value,
        penalty=penalty,
    )

    scores = cross_val_score(
        model,
        X_development,
        y_development,
        scoring="roc_auc",
        cv=search_cv,
        n_jobs=-1,
    )

    return float(scores.mean())


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    ),
)

optuna_started = time.perf_counter()

study.optimize(
    optuna_objective,
    n_trials=8,
)

optuna_time = time.perf_counter() - optuna_started


# ============================================================
# Сравнение методов поиска
# ============================================================

search_methods = pd.DataFrame(
    [
        {
            "method": "Grid Search",
            "iterations": len(grid_search.cv_results_["params"]),
            "best_cv_roc_auc": grid_search.best_score_,
            "time_sec": grid_time,
            "best_parameters": str(grid_search.best_params_),
        },
        {
            "method": "Random Search",
            "iterations": len(random_search.cv_results_["params"]),
            "best_cv_roc_auc": random_search.best_score_,
            "time_sec": random_time,
            "best_parameters": str(
                random_search.best_params_
            ),
        },
        {
            "method": "Optuna",
            "iterations": len(study.trials),
            "best_cv_roc_auc": study.best_value,
            "time_sec": optuna_time,
            "best_parameters": str(study.best_params),
        },
    ]
)

search_methods.to_csv(
    HERE / "advanced_search_comparison.csv",
    index=False,
)

print("\n=== Сравнение методов поиска ===")
print(
    search_methods.to_string(
        index=False,
        formatters={
            "best_cv_roc_auc": "{:.4f}".format,
            "time_sec": "{:.3f}".format,
        },
    )
)


# ============================================================
# График методов поиска
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
)

axes[0].bar(
    search_methods["method"],
    search_methods["best_cv_roc_auc"],
    color=[
        "#457B9D",
        "#E76F51",
        "#2A9D8F",
    ],
)

axes[0].set_title("Лучший ROC-AUC")
axes[0].set_ylabel("ROC-AUC")
axes[0].set_ylim(
    max(
        0.5,
        search_methods["best_cv_roc_auc"].min() - 0.03,
    ),
    min(
        1.0,
        search_methods["best_cv_roc_auc"].max() + 0.03,
    ),
)
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(
    search_methods["method"],
    search_methods["time_sec"],
    color=[
        "#457B9D",
        "#E76F51",
        "#2A9D8F",
    ],
)

axes[1].set_title("Время поиска")
axes[1].set_ylabel("Секунды")
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    HERE / "advanced_search_comparison.png",
    dpi=160,
)
plt.close()


# ============================================================
# Сравнение L1 и L2
# ============================================================

regularization_rows = []

for penalty in ["l1", "l2"]:
    penalty_search = GridSearchCV(
        estimator=make_model(penalty=penalty),
        param_grid={
            "model__C": [
                0.01,
                0.1,
                1.0,
                10.0,
            ]
        },
        scoring="roc_auc",
        cv=search_cv,
        n_jobs=-1,
        refit=True,
    )

    penalty_search.fit(
        X_development,
        y_development,
    )

    coefficients = (
        penalty_search
        .best_estimator_
        .named_steps["model"]
        .coef_
        .ravel()
    )

    nonzero = int(
        np.count_nonzero(
            np.abs(coefficients) > 1e-8
        )
    )

    total_coefficients = len(coefficients)

    regularization_rows.append(
        {
            "penalty": penalty.upper(),
            "best_C": penalty_search.best_params_[
                "model__C"
            ],
            "best_cv_roc_auc": penalty_search.best_score_,
            "nonzero_coefficients": nonzero,
            "total_coefficients": total_coefficients,
        }
    )

regularization_results = pd.DataFrame(
    regularization_rows
)

regularization_results.to_csv(
    HERE / "regularization_results.csv",
    index=False,
)

print("\n=== L1 против L2 ===")
print(
    regularization_results.to_string(
        index=False,
        formatters={
            "best_cv_roc_auc": "{:.4f}".format,
        },
    )
)


# ============================================================
# График регуляризации
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.5),
)

axes[0].bar(
    regularization_results["penalty"],
    regularization_results["best_cv_roc_auc"],
    color=[
        "#E9C46A",
        "#264653",
    ],
)

axes[0].set_title("Качество L1 и L2")
axes[0].set_ylabel("ROC-AUC")
axes[0].set_ylim(
    max(
        0.5,
        regularization_results[
            "best_cv_roc_auc"
        ].min() - 0.03,
    ),
    min(
        1.0,
        regularization_results[
            "best_cv_roc_auc"
        ].max() + 0.03,
    ),
)
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(
    regularization_results["penalty"],
    regularization_results[
        "nonzero_coefficients"
    ],
    color=[
        "#E9C46A",
        "#264653",
    ],
)

axes[1].set_title("Ненулевые коэффициенты")
axes[1].set_ylabel("Количество")
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    HERE / "regularization_comparison.png",
    dpi=160,
)
plt.close()


# ============================================================
# Вывод
# ============================================================

print("\n=== Практическая рекомендация ===")

print(
    "Вложенная кросс-валидация оправдана, когда датасет "
    "небольшой, проводится большой подбор параметров и нужна "
    "максимально честная оценка качества."
)

print(
    "Для больших данных или быстрого прототипа её стоимость "
    "может быть слишком высокой. Тогда достаточно отдельной "
    "test-выборки и обычного GridSearchCV только на train."
)

print(
    "L1 может занулять коэффициенты и выполнять отбор признаков. "
    "L2 обычно сохраняет все признаки и плавно уменьшает веса."
)

print("\n=== Созданные файлы ===")
print("nested_cv_results.csv")
print("nested_cv_comparison.png")
print("advanced_search_comparison.csv")
print("advanced_search_comparison.png")
print("regularization_results.csv")
print("regularization_comparison.png")


def main():
    """Все вычисления выполняются при запуске файла."""
    pass


if __name__ == "__main__":
    main()
