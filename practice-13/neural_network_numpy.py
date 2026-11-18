from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(".")
RANDOM_STATE = 42


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class NumPyMLP:
    def __init__(self, input_size, hidden_size=16,
                 activation="relu", random_state=42):
        self.activation_name = activation
        rng = np.random.default_rng(random_state)

        if activation == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        self.W1 = rng.normal(
            0, scale, size=(input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(
            0, np.sqrt(1.0 / hidden_size),
            size=(hidden_size, 1),
        )
        self.b2 = np.zeros((1, 1))
        self.loss_curve_ = []

    def activation(self, z):
        if self.activation_name == "relu":
            return np.maximum(0, z)
        if self.activation_name == "tanh":
            return np.tanh(z)
        raise ValueError("Неизвестная функция активации")

    def activation_derivative(self, z):
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        if self.activation_name == "tanh":
            return 1.0 - np.tanh(z) ** 2
        raise ValueError("Неизвестная функция активации")

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.activation(z1)
        z2 = a1 @ self.W2 + self.b2
        probability = sigmoid(z2)
        return z1, a1, probability

    def loss(self, X, y):
        _, _, probability = self.forward(X)
        y_column = y.reshape(-1, 1)
        eps = 1e-12

        return -np.mean(
            y_column * np.log(probability + eps)
            + (1 - y_column) * np.log(1 - probability + eps)
        )

    def loss_and_gradients(self, X, y):
        n = len(X)
        y_column = y.reshape(-1, 1)

        z1, a1, probability = self.forward(X)

        dz2 = (probability - y_column) / n
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.activation_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

        return self.loss(X, y), gradients

    def fit(self, X, y, epochs=1500, learning_rate=0.05):
        self.loss_curve_ = []

        for _ in range(epochs):
            loss, gradients = self.loss_and_gradients(X, y)
            self.loss_curve_.append(loss)

            self.W1 -= learning_rate * gradients["W1"]
            self.b1 -= learning_rate * gradients["b1"]
            self.W2 -= learning_rate * gradients["W2"]
            self.b2 -= learning_rate * gradients["b2"]

        return self

    def predict_proba(self, X):
        _, _, probability = self.forward(X)
        return probability.ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def gradient_check():
    rng = np.random.default_rng(42)
    X_small = rng.normal(size=(8, 2))
    y_small = rng.integers(0, 2, size=8)

    model = NumPyMLP(
        input_size=2,
        hidden_size=4,
        activation="tanh",
        random_state=42,
    )

    _, analytical = model.loss_and_gradients(X_small, y_small)

    epsilon = 1e-5
    rows = []

    for name in ["W1", "b1", "W2", "b2"]:
        parameter = getattr(model, name)
        gradient = analytical[name]
        errors = []

        for index in np.ndindex(parameter.shape):
            old_value = parameter[index]

            parameter[index] = old_value + epsilon
            loss_plus = model.loss(X_small, y_small)

            parameter[index] = old_value - epsilon
            loss_minus = model.loss(X_small, y_small)

            parameter[index] = old_value

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical_value = gradient[index]

            denominator = max(
                1e-8,
                abs(numerical) + abs(analytical_value),
            )
            relative_error = (
                abs(numerical - analytical_value) / denominator
            )
            errors.append(relative_error)

        rows.append({
            "parameter": name,
            "max_relative_error": max(errors),
        })

    return pd.DataFrame(rows)


def draw_boundary(ax, score_function, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )

    grid = np.column_stack([xx.ravel(), yy.ravel()])
    scores = score_function(grid).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        scores,
        levels=np.linspace(0, 1, 20),
        cmap="coolwarm",
        alpha=0.35,
    )
    ax.contour(
        xx,
        yy,
        scores,
        levels=[0.5],
        colors="black",
        linewidths=1.5,
    )
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        s=18,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_title(title)
    ax.grid(alpha=0.2)


# Двумерные данные для наглядной проверки
X, y = make_moons(
    n_samples=600,
    noise=0.22,
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=== Данные ===")
print("Train:", X_train_scaled.shape)
print("Test:", X_test_scaled.shape)
print("Классы train:", np.bincount(y_train))


# Проверка правильности backpropagation
gradient_results = gradient_check()
gradient_results.to_csv(
    OUTPUT_DIR / "gradient_check.csv",
    index=False,
)

print("\n=== Численная проверка градиентов ===")
print(gradient_results.to_string(index=False))

max_gradient_error = gradient_results[
    "max_relative_error"
].max()

print("Максимальная относительная ошибка:", max_gradient_error)

if max_gradient_error < 1e-5:
    print("Проверка пройдена: градиенты реализованы правильно.")
else:
    print("Внимание: необходимо проверить формулы градиентов.")


results = []
trained_models = {}

for activation_name in ["relu", "tanh"]:
    numpy_model = NumPyMLP(
        input_size=2,
        hidden_size=16,
        activation=activation_name,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()
    numpy_model.fit(
        X_train_scaled,
        y_train,
        epochs=1500,
        learning_rate=0.05,
    )
    elapsed = time.perf_counter() - start

    probability = numpy_model.predict_proba(X_test_scaled)
    prediction = numpy_model.predict(X_test_scaled)

    results.append({
        "implementation": "NumPy",
        "activation": activation_name,
        "accuracy_test": accuracy_score(y_test, prediction),
        "roc_auc_test": roc_auc_score(y_test, probability),
        "fit_time_sec": elapsed,
        "epochs": len(numpy_model.loss_curve_),
        "final_loss": numpy_model.loss_curve_[-1],
    })

    trained_models[("NumPy", activation_name)] = numpy_model

    sklearn_model = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation=activation_name,
        solver="adam",
        learning_rate_init=0.01,
        max_iter=1500,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()
    sklearn_model.fit(X_train_scaled, y_train)
    elapsed = time.perf_counter() - start

    probability = sklearn_model.predict_proba(X_test_scaled)[:, 1]
    prediction = sklearn_model.predict(X_test_scaled)

    results.append({
        "implementation": "sklearn",
        "activation": activation_name,
        "accuracy_test": accuracy_score(y_test, prediction),
        "roc_auc_test": roc_auc_score(y_test, probability),
        "fit_time_sec": elapsed,
        "epochs": sklearn_model.n_iter_,
        "final_loss": sklearn_model.loss_curve_[-1],
    })

    trained_models[("sklearn", activation_name)] = sklearn_model


results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "numpy_network_results.csv",
    index=False,
)

print("\n=== NumPy против sklearn ===")
print(
    results_df.to_string(
        index=False,
        formatters={
            "accuracy_test": "{:.4f}".format,
            "roc_auc_test": "{:.4f}".format,
            "fit_time_sec": "{:.4f}".format,
            "final_loss": "{:.6f}".format,
        },
    )
)


# График сходимости
plt.figure(figsize=(9, 5))

for key, model in trained_models.items():
    implementation, activation_name = key

    plt.plot(
        model.loss_curve_,
        label=f"{implementation}, {activation_name}",
    )

plt.xlabel("Эпоха")
plt.ylabel("Log loss")
plt.title("Сходимость нейронных сетей")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "numpy_loss_comparison.png",
    dpi=160,
)
plt.close()


# Границы решений
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for ax, key in zip(axes.ravel(), trained_models):
    implementation, activation_name = key
    model = trained_models[key]

    if implementation == "NumPy":
        score_function = model.predict_proba
    else:
        score_function = lambda data, m=model: (
            m.predict_proba(data)[:, 1]
        )

    draw_boundary(
        ax,
        score_function,
        X_test_scaled,
        y_test,
        f"{implementation}, {activation_name}",
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "numpy_decision_boundaries.png",
    dpi=160,
)
plt.close()


best_row = results_df.loc[
    results_df["roc_auc_test"].idxmax()
]

print("\n=== Вывод ===")
print(
    "Лучший вариант:",
    best_row["implementation"],
    best_row["activation"],
)
print("ROC-AUC:", round(best_row["roc_auc_test"], 4))
print(
    "Ручная реализация выполняет прямое распространение, "
    "backpropagation и обновление весов без sklearn."
)
print(
    "Численная проверка подтверждает правильность "
    "аналитических градиентов."
)
print(
    "ReLU и tanh могут сходиться с разной скоростью, "
    "поэтому выбор активации проверяется экспериментально."
)

print("\n=== Созданные файлы ===")
print("gradient_check.csv")
print("numpy_network_results.csv")
print("numpy_loss_comparison.png")
print("numpy_decision_boundaries.png")
