#!/usr/bin/env python3
"""Практическая работа № 8: градиентный бустинг для прогноза оттока."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "o_churn_synthetic.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def make_preprocessor(*, scale_numeric: bool = False) -> ColumnTransformer:
    """One-hot для категорий; числа пропускаются или масштабируются."""
    remainder = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["region", "tariff"],
            )
        ],
        remainder=remainder,
    )


def evaluate(name, pipeline, x_train, y_train, x_test, y_test) -> dict:
    """Обучить модель и вернуть одинаковый набор метрик."""
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    seconds = time.perf_counter() - started

    train_prediction = pipeline.predict(x_train)
    test_prediction = pipeline.predict(x_test)
    test_probability = pipeline.predict_proba(x_test)[:, 1]

    return {
        "model": name,
        "f1_train": f1_score(y_train, train_prediction),
        "f1_test": f1_score(y_test, test_prediction),
        "roc_auc_test": roc_auc_score(y_test, test_probability),
        "training_seconds": seconds,
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    x = data.drop(columns=["churn", "subscriber_code"])
    y = data["churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("=== Окружение ===")
    print(f"pandas: {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"xgboost: {xgboost.__version__}")

    print("\n=== Данные и единое разбиение ===")
    print(f"Файл: {DATA_PATH}")
    print(f"Датасет: {data.shape[0]} строк, {data.shape[1]} столбцов")
    print(f"Доля churn=1: {y.mean():.4f}")
    print(f"Train: {x_train.shape}, test: {x_test.shape}")
    print(f"random_state: {RANDOM_STATE}")

    # Базовое сравнение со слайда: лес и XGBoost на одном split.
    baseline_models = {
        "Random forest, 100": make_pipeline(
            make_preprocessor(),
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "XGBoost template": make_pipeline(
            make_preprocessor(),
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }
    baseline_rows = [
        evaluate(name, model, x_train, y_train, x_test, y_test)
        for name, model in baseline_models.items()
    ]
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(HERE / "baseline_metrics.csv", index=False)
    print("\n=== Случайный лес и шаблон XGBoost ===")
    print(baseline.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Валидация отделяется от train. Test остаётся нетронутым.
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    validation_preprocessor = make_preprocessor()
    xtr = validation_preprocessor.fit_transform(x_tr)
    xval = validation_preprocessor.transform(x_val)
    print("\n=== Validation split без утечки ===")
    print(
        f"X_tr: {len(x_tr)}, X_val: {len(x_val)}, X_test: {len(x_test)}, "
        f"признаков после one-hot: {xtr.shape[1]}"
    )

    # Подбор learning_rate и max_depth с ранней остановкой.
    search_rows: list[dict] = []
    best_row: dict | None = None
    print("\n=== Перебор learning_rate и max_depth ===")
    for learning_rate in [0.3, 0.1, 0.05, 0.01]:
        for max_depth in [3, 4, 6]:
            model = XGBClassifier(
                n_estimators=3000,
                learning_rate=learning_rate,
                max_depth=max_depth,
                eval_metric="logloss",
                early_stopping_rounds=50,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            started = time.perf_counter()
            model.fit(
                xtr,
                y_tr,
                eval_set=[(xval, y_val)],
                verbose=False,
            )
            seconds = time.perf_counter() - started
            probability = model.predict_proba(xval)[:, 1]
            row = {
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "trees": model.best_iteration + 1,
                "best_logloss": model.best_score,
                "roc_auc_validation": roc_auc_score(y_val, probability),
                "training_seconds": seconds,
            }
            search_rows.append(row)
            if best_row is None or row["roc_auc_validation"] > best_row["roc_auc_validation"]:
                best_row = row
            print(
                f"lr={learning_rate:<4} depth={max_depth} "
                f"trees={row['trees']:<4} "
                f"AUC val={row['roc_auc_validation']:.4f}"
            )

    assert best_row is not None
    search = pd.DataFrame(search_rows)
    search.to_csv(HERE / "xgboost_search_results.csv", index=False)
    print("\nЛучшие параметры по validation ROC-AUC:")
    print(
        f"learning_rate={best_row['learning_rate']}, "
        f"max_depth={best_row['max_depth']}, trees={best_row['trees']}, "
        f"AUC={best_row['roc_auc_validation']:.4f}"
    )

    # Финальное сравнение. Все модели получают один и тот же train/test.
    final_models = {
        "Logistic regression": make_pipeline(
            make_preprocessor(scale_numeric=True),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Random forest": make_pipeline(
            make_preprocessor(),
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "XGBoost tuned": make_pipeline(
            make_preprocessor(),
            XGBClassifier(
                n_estimators=int(best_row["trees"]),
                learning_rate=float(best_row["learning_rate"]),
                max_depth=int(best_row["max_depth"]),
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }
    final_rows = [
        evaluate(name, model, x_train, y_train, x_test, y_test)
        for name, model in final_models.items()
    ]
    final_metrics = pd.DataFrame(final_rows)
    final_metrics.to_csv(HERE / "final_model_metrics.csv", index=False)
    print("\n=== Итоговое сравнение на одном test split ===")
    print(final_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    summary = {
        "data_rows": len(data),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "validation_rows": len(x_val),
        "random_state": RANDOM_STATE,
        "xgboost_version": xgboost.__version__,
        "best_xgboost": {
            "learning_rate": float(best_row["learning_rate"]),
            "max_depth": int(best_row["max_depth"]),
            "n_estimators": int(best_row["trees"]),
            "validation_roc_auc": float(best_row["roc_auc_validation"]),
        },
        "final_metrics": final_rows,
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Чек-лист ===")
    print("[OK] Все модели используют один split и random_state=42")
    print("[OK] Таблица содержит F1 train, F1 test, ROC-AUC test и время")
    print(f"[OK] Версия xgboost указана: {xgboost.__version__}")
    print("[OK] Test не использовался для подбора параметров")


if __name__ == "__main__":
    main()
