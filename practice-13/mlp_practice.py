from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "o_churn_synthetic.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_preprocessor(scale_numeric=True):
    categorical_columns = [
        "region",
        "tariff",
    ]

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "numeric",
                numeric_transformer,
                numeric_columns,
            ),
        ]
    )


def make_mlp(
    hidden_layers,
    random_state=RANDOM_STATE,
):
    return Pipeline(
        steps=[
            (
                "preprocess",
                make_preprocessor(
                    scale_numeric=True
                ),
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.20,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_xgboost(
    random_state=RANDOM_STATE,
):
    return Pipeline(
        steps=[
            (
                "preprocess",
                make_preprocessor(
                    scale_numeric=False
                ),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=70,
                    learning_rate=0.1,
                    max_depth=3,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(
    name,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
):
    started = time.perf_counter()

    model.fit(
        x_train,
        y_train,
    )

    elapsed = time.perf_counter() - started

    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    train_probability = (
        model.predict_proba(x_train)[:, 1]
    )

    test_probability = (
        model.predict_proba(x_test)[:, 1]
    )

    result = {
        "model": name,
        "accuracy_train": accuracy_score(
            y_train,
            train_prediction,
        ),
        "accuracy_test": accuracy_score(
            y_test,
            test_prediction,
        ),
        "f1_test": f1_score(
            y_test,
            test_prediction,
        ),
        "roc_auc_train": roc_auc_score(
            y_train,
            train_probability,
        ),
        "roc_auc_test": roc_auc_score(
            y_test,
            test_probability,
        ),
        "fit_time_sec": elapsed,
    }

    return result, model


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

X_development, X_test, y_development, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
)

print("=== Данные ===")
print("Объектов:", len(data))
print("Признаков:", X.shape[1])
print("Пропусков:", int(data.isna().sum().sum()))
print("Development:", X_development.shape)
print("Test:", X_test.shape)
print("Доля churn=1:", round(float(y.mean()), 4))


# ============================================================
# Уровни A и B: подбор архитектуры только на development
# ============================================================

architectures = [
    (16,),
    (32,),
    (64,),
    (32, 16),
    (64, 32),
]

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)

architecture_rows = []

print("\n=== Подбор архитектуры MLP ===")

for architecture in architectures:
    pipeline = make_mlp(
        hidden_layers=architecture,
    )

    started = time.perf_counter()

    scores = cross_val_score(
        pipeline,
        X_development,
        y_development,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    elapsed = time.perf_counter() - started

    row = {
        "architecture": str(architecture),
        "layers": len(architecture),
        "neurons_total": sum(architecture),
        "cv_roc_auc_mean": scores.mean(),
        "cv_roc_auc_std": scores.std(),
        "search_time_sec": elapsed,
    }

    architecture_rows.append(row)

    print(
        f"{str(architecture):<12} "
        f"ROC-AUC={scores.mean():.4f} "
        f"± {scores.std():.4f}, "
        f"время={elapsed:.3f} с"
    )

architecture_results = pd.DataFrame(
    architecture_rows
)

architecture_results.to_csv(
    HERE / "architecture_results.csv",
    index=False,
)

best_index = (
    architecture_results[
        "cv_roc_auc_mean"
    ].idxmax()
)

best_architecture_text = (
    architecture_results.loc[
        best_index,
        "architecture",
    ]
)

best_architecture = eval(
    best_architecture_text
)

print("\nЛучшая архитектура:", best_architecture)
print(
    "Лучший ROC-AUC CV:",
    round(
        float(
            architecture_results.loc[
                best_index,
                "cv_roc_auc_mean",
            ]
        ),
        4,
    ),
)


# ============================================================
# График архитектур
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
)

architecture_labels = (
    architecture_results[
        "architecture"
    ]
)

axes[0].bar(
    architecture_labels,
    architecture_results[
        "cv_roc_auc_mean"
    ],
    yerr=architecture_results[
        "cv_roc_auc_std"
    ],
    capsize=5,
    color="#457B9D",
)

axes[0].set_title("ROC-AUC разных архитектур")
axes[0].set_ylabel("ROC-AUC CV")
axes[0].tick_params(
    axis="x",
    rotation=30,
)
axes[0].grid(
    axis="y",
    alpha=0.25,
)

axes[1].bar(
    architecture_labels,
    architecture_results[
        "search_time_sec"
    ],
    color="#E76F51",
)

axes[1].set_title("Время кросс-валидации")
axes[1].set_ylabel("Секунды")
axes[1].tick_params(
    axis="x",
    rotation=30,
)
axes[1].grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    HERE / "architecture_comparison.png",
    dpi=160,
)

plt.close()


# ============================================================
# Обучение лучшей MLP
# ============================================================

best_mlp = make_mlp(
    hidden_layers=best_architecture,
)

mlp_result, best_mlp = evaluate_model(
    name="MLP",
    model=best_mlp,
    x_train=X_development,
    y_train=y_development,
    x_test=X_test,
    y_test=y_test,
)

trained_mlp = (
    best_mlp.named_steps["model"]
)

mlp_result["epochs"] = trained_mlp.n_iter_
mlp_result["architecture"] = str(
    best_architecture
)

print("\n=== Лучшая MLP ===")
print("Архитектура:", best_architecture)
print("Эпох выполнено:", trained_mlp.n_iter_)
print(
    "Ранняя остановка:",
    trained_mlp.n_iter_ < trained_mlp.max_iter,
)
print(
    "Accuracy test:",
    round(
        mlp_result["accuracy_test"],
        4,
    ),
)
print(
    "F1 test:",
    round(
        mlp_result["f1_test"],
        4,
    ),
)
print(
    "ROC-AUC test:",
    round(
        mlp_result["roc_auc_test"],
        4,
    ),
)


# ============================================================
# График функции потерь
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    np.arange(
        1,
        len(trained_mlp.loss_curve_) + 1,
    ),
    trained_mlp.loss_curve_,
    color="#264653",
    linewidth=2,
)

plt.xlabel("Эпоха")
plt.ylabel("Log loss")
plt.title(
    "Кривая потерь лучшей MLP\n"
    f"Архитектура {best_architecture}"
)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    HERE / "mlp_loss_curve.png",
    dpi=160,
)

plt.close()


# ============================================================
# XGBoost из практической работы №8
# ============================================================

xgb_result, trained_xgb = evaluate_model(
    name="XGBoost",
    model=make_xgboost(),
    x_train=X_development,
    y_train=y_development,
    x_test=X_test,
    y_test=y_test,
)

xgb_result["epochs"] = 70
xgb_result["architecture"] = (
    "70 деревьев, depth=3"
)


# ============================================================
# Итоговое сравнение
# ============================================================

model_comparison = pd.DataFrame(
    [
        mlp_result,
        xgb_result,
    ]
)

model_comparison.to_csv(
    HERE / "model_comparison.csv",
    index=False,
)

print("\n=== MLP против XGBoost ===")
print(
    model_comparison.to_string(
        index=False,
        formatters={
            "accuracy_train": "{:.4f}".format,
            "accuracy_test": "{:.4f}".format,
            "f1_test": "{:.4f}".format,
            "roc_auc_train": "{:.4f}".format,
            "roc_auc_test": "{:.4f}".format,
            "fit_time_sec": "{:.4f}".format,
        },
    )
)


# ============================================================
# График сравнения моделей
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
)

x_positions = np.arange(
    len(model_comparison)
)

width = 0.35

axes[0].bar(
    x_positions - width / 2,
    model_comparison["f1_test"],
    width,
    label="F1",
    color="#2A9D8F",
)

axes[0].bar(
    x_positions + width / 2,
    model_comparison["roc_auc_test"],
    width,
    label="ROC-AUC",
    color="#E9C46A",
)

axes[0].set_xticks(
    x_positions,
    model_comparison["model"],
)

axes[0].set_ylim(0.4, 1.0)
axes[0].set_title("Качество на test")
axes[0].legend()
axes[0].grid(
    axis="y",
    alpha=0.25,
)

axes[1].bar(
    model_comparison["model"],
    model_comparison["fit_time_sec"],
    color=[
        "#457B9D",
        "#E76F51",
    ],
)

axes[1].set_title("Время обучения")
axes[1].set_ylabel("Секунды")
axes[1].grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    HERE / "model_comparison.png",
    dpi=160,
)

plt.close()


# ============================================================
# Стабильность при разных random_state
# ============================================================

random_states = [
    1,
    7,
    21,
    42,
    99,
]

stability_rows = []

print("\n=== Стабильность моделей ===")

for state in random_states:
    models = {
        "MLP": make_mlp(
            hidden_layers=best_architecture,
            random_state=state,
        ),
        "XGBoost": make_xgboost(
            random_state=state,
        ),
    }

    for model_name, model in models.items():
        started = time.perf_counter()

        model.fit(
            X_development,
            y_development,
        )

        elapsed = time.perf_counter() - started

        prediction = model.predict(X_test)
        probability = (
            model.predict_proba(X_test)[:, 1]
        )

        row = {
            "model": model_name,
            "random_state": state,
            "f1_test": f1_score(
                y_test,
                prediction,
            ),
            "roc_auc_test": roc_auc_score(
                y_test,
                probability,
            ),
            "fit_time_sec": elapsed,
        }

        stability_rows.append(row)

        print(
            f"{model_name:<8} "
            f"state={state:<3} "
            f"F1={row['f1_test']:.4f} "
            f"AUC={row['roc_auc_test']:.4f}"
        )

stability_results = pd.DataFrame(
    stability_rows
)

stability_results.to_csv(
    HERE / "stability_results.csv",
    index=False,
)


# ============================================================
# Сводка стабильности
# ============================================================

stability_summary = (
    stability_results
    .groupby("model")
    .agg(
        roc_auc_mean=(
            "roc_auc_test",
            "mean",
        ),
        roc_auc_std=(
            "roc_auc_test",
            "std",
        ),
        f1_mean=(
            "f1_test",
            "mean",
        ),
        f1_std=(
            "f1_test",
            "std",
        ),
        time_mean=(
            "fit_time_sec",
            "mean",
        ),
    )
    .reset_index()
)

stability_summary.to_csv(
    HERE / "stability_summary.csv",
    index=False,
)

print("\n=== Средняя стабильность ===")
print(
    stability_summary.to_string(
        index=False,
        formatters={
            "roc_auc_mean": "{:.4f}".format,
            "roc_auc_std": "{:.4f}".format,
            "f1_mean": "{:.4f}".format,
            "f1_std": "{:.4f}".format,
            "time_mean": "{:.4f}".format,
        },
    )
)


# ============================================================
# Boxplot стабильности
# ============================================================

mlp_auc = stability_results.loc[
    stability_results["model"] == "MLP",
    "roc_auc_test",
]

xgb_auc = stability_results.loc[
    stability_results["model"] == "XGBoost",
    "roc_auc_test",
]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        mlp_auc,
        xgb_auc,
    ],
    tick_labels=[
        "MLP",
        "XGBoost",
    ],
)

plt.ylabel("ROC-AUC test")
plt.title(
    "Стабильность при разных random_state"
)
plt.grid(
    axis="y",
    alpha=0.25,
)
plt.tight_layout()

plt.savefig(
    HERE / "stability_comparison.png",
    dpi=160,
)

plt.close()


# ============================================================
# Вывод
# ============================================================

mlp_auc_mean = float(
    stability_summary.loc[
        stability_summary["model"] == "MLP",
        "roc_auc_mean",
    ].iloc[0]
)

xgb_auc_mean = float(
    stability_summary.loc[
        stability_summary["model"] == "XGBoost",
        "roc_auc_mean",
    ].iloc[0]
)

print("\n=== Итоговый вывод ===")

if mlp_auc_mean > xgb_auc_mean:
    print(
        "MLP показала более высокий средний ROC-AUC, "
        "но необходимо учитывать время и стабильность."
    )
else:
    print(
        "XGBoost показал не хуже или лучше средний "
        "ROC-AUC при меньшей сложности настройки."
    )

print(
    "На табличных данных умеренного размера нейронная "
    "сеть не гарантирует преимущество перед бустингом."
)

print(
    "Применение MLP оправдано только при подтверждённом "
    "выигрыше на честной валидации."
)

print("\n=== Созданные файлы ===")
print("architecture_results.csv")
print("architecture_comparison.png")
print("mlp_loss_curve.png")
print("model_comparison.csv")
print("model_comparison.png")
print("stability_results.csv")
print("stability_summary.csv")
print("stability_comparison.png")


def main():
    """Вычисления выполняются при запуске файла."""
    pass


if __name__ == "__main__":
    main()
