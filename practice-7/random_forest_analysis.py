#!/usr/bin/env python3
"""Практическая №7: дерево решений и случайный лес на Telco Churn."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "practice-5" / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    target = data.pop("Churn").map({"No": 0, "Yes": 1})
    data = data.drop(columns=["customerID"])
    return data, target


def columns_for(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = data.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in data.columns if column not in numeric]
    return numeric, categorical


def make_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = columns_for(data)
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def make_pipeline(data: pd.DataFrame, classifier) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(data)),
            ("model", classifier),
        ]
    )


def metrics_for(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "tn": int(confusion_matrix(y_test, prediction)[0, 0]),
        "fp": int(confusion_matrix(y_test, prediction)[0, 1]),
        "fn": int(confusion_matrix(y_test, prediction)[1, 0]),
        "tp": int(confusion_matrix(y_test, prediction)[1, 1]),
    }


def aggregate_importance(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    forest = model.named_steps["model"]
    numeric, categorical = columns_for(data)
    importances = forest.feature_importances_
    result: list[tuple[str, float]] = []

    numeric_slice = preprocessor.output_indices_["num"]
    for offset, column in enumerate(numeric):
        result.append((column, float(importances[numeric_slice.start + offset])))

    categorical_slice = preprocessor.output_indices_["cat"]
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cursor = categorical_slice.start
    for column, categories in zip(categorical, onehot.categories_):
        next_cursor = cursor + len(categories)
        result.append((column, float(importances[cursor:next_cursor].sum())))
        cursor = next_cursor

    frame = pd.DataFrame(result, columns=["feature", "importance"])
    return frame.sort_values("importance", ascending=False, ignore_index=True)


def plot_importance(frame: pd.DataFrame) -> None:
    shown = frame.sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 8))
    plt.barh(shown["feature"], shown["importance"], color="#2878B5")
    plt.xlabel("Суммарная impurity importance")
    plt.ylabel("Исходный признак")
    plt.title("Важность признаков настроенного случайного леса")
    plt.tight_layout()
    plt.savefig(HERE / "feature_importance.png", dpi=170)
    plt.close()


def main() -> None:
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    x, y = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=== Данные ===")
    print(f"Файл: {data_path}")
    print(f"Объектов: {len(x)}, исходных признаков: {x.shape[1]}")
    print(f"Train: {len(x_train)}, test: {len(x_test)}")
    print(f"Доля Churn=Yes: {y.mean():.4f}")

    tree = make_pipeline(
        x_train,
        DecisionTreeClassifier(random_state=RANDOM_STATE),
    )
    default_forest = make_pipeline(
        x_train,
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    )
    tree.fit(x_train, y_train)
    default_forest.fit(x_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search_pipeline = make_pipeline(
        x_train,
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
    )
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [8, 14, None],
        "model__min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        search_pipeline,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        verbose=1,
    )
    print("\n=== GridSearchCV ===")
    print(f"Комбинаций: {len(search.get_params()['param_grid']['model__n_estimators']) * len(search.get_params()['param_grid']['model__max_depth']) * len(search.get_params()['param_grid']['model__min_samples_leaf'])}")
    search.fit(x_train, y_train)
    print(f"Лучшие параметры: {search.best_params_}")
    print(f"Лучший средний CV ROC-AUC: {search.best_score_:.4f}")

    best = search.best_params_
    tuned_forest = make_pipeline(
        x_train,
        RandomForestClassifier(
            n_estimators=best["model__n_estimators"],
            max_depth=best["model__max_depth"],
            min_samples_leaf=best["model__min_samples_leaf"],
            class_weight="balanced",
            oob_score=True,
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    tuned_forest.fit(x_train, y_train)

    results = [
        metrics_for("Decision tree", tree, x_test, y_test),
        metrics_for("Random forest default", default_forest, x_test, y_test),
        metrics_for("Random forest tuned", tuned_forest, x_test, y_test),
    ]
    metrics = pd.DataFrame(results)
    metrics.to_csv(HERE / "model_metrics.csv", index=False)

    transformed_shape = tuned_forest.named_steps["preprocess"].transform(x_train).shape
    oob = tuned_forest.named_steps["model"].oob_score_
    print("\n=== Сравнение на одном test split ===")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nРазмерность после preprocessing: {transformed_shape}")
    print(f"OOB accuracy настроенного леса: {oob:.4f}")

    importance = aggregate_importance(tuned_forest, x_train)
    importance.to_csv(HERE / "feature_importance.csv", index=False)
    plot_importance(importance)
    print("\n=== Важность исходных признаков ===")
    print(importance.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    removal_results = []
    least_first = importance.sort_values("importance", ascending=True)["feature"].tolist()
    for remove_count in [0, 1, 3, 5, 7]:
        removed = least_first[:remove_count]
        keep = [column for column in x.columns if column not in removed]
        reduced_train = x_train[keep]
        reduced_test = x_test[keep]
        reduced_model = make_pipeline(
            reduced_train,
            RandomForestClassifier(
                n_estimators=best["model__n_estimators"],
                max_depth=best["model__max_depth"],
                min_samples_leaf=best["model__min_samples_leaf"],
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
        reduced_model.fit(reduced_train, y_train)
        row = metrics_for(
            f"remove_{remove_count}", reduced_model, reduced_test, y_test
        )
        row["removed_count"] = remove_count
        row["removed_features"] = ", ".join(removed) if removed else "—"
        removal_results.append(row)

    removal = pd.DataFrame(removal_results)
    baseline_auc = removal.loc[removal["removed_count"] == 0, "roc_auc"].iloc[0]
    baseline_f1 = removal.loc[removal["removed_count"] == 0, "f1"].iloc[0]
    removal["roc_auc_delta"] = removal["roc_auc"] - baseline_auc
    removal["f1_delta"] = removal["f1"] - baseline_f1
    removal.to_csv(HERE / "feature_removal_experiment.csv", index=False)
    print("\n=== Удаление наименее важных признаков ===")
    print(
        removal[
            [
                "removed_count",
                "removed_features",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "f1_delta",
                "roc_auc_delta",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results[
        [
            "rank_test_score",
            "param_model__n_estimators",
            "param_model__max_depth",
            "param_model__min_samples_leaf",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
        ]
    ].to_csv(HERE / "grid_search_results.csv", index=False)

    summary = {
        "best_params": {
            "n_estimators": int(best["model__n_estimators"]),
            "max_depth": best["model__max_depth"],
            "min_samples_leaf": int(best["model__min_samples_leaf"]),
        },
        "best_cv_roc_auc": float(search.best_score_),
        "oob_accuracy": float(oob),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "original_features": x.shape[1],
        "transformed_features": transformed_shape[1],
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nСозданы CSV, summary.json и feature_importance.png")


if __name__ == "__main__":
    main()
