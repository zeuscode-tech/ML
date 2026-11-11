import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. Создание двумерного датасета
X, y = make_moons(
    n_samples=300,
    noise=0.25,
    random_state=42
)

# 2. Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Количество объектов:", len(X_scaled))
print("Количество признаков:", X_scaled.shape[1])
print("Распределение классов:", np.bincount(y))
print("Средние после масштабирования:",
      np.round(X_scaled.mean(axis=0), 3))
print("Стандартные отклонения:",
      np.round(X_scaled.std(axis=0), 3))


def plot_boundary(model, X, y, title, filename, axis):
    x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
    y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    prediction = model.predict(grid).reshape(xx.shape)

    axis.contourf(
        xx,
        yy,
        prediction,
        alpha=0.25,
        cmap="coolwarm"
    )

    axis.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="black",
        s=30
    )

    # Опорные векторы изображаются большими пустыми окружностями
    support = model.support_vectors_
    axis.scatter(
        support[:, 0],
        support[:, 1],
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.3,
        label="Опорные векторы"
    )

    accuracy = accuracy_score(y, model.predict(X))

    axis.set_title(
        f"{title}\nAccuracy={accuracy:.3f}, "
        f"опорных векторов={len(support)}"
    )
    axis.set_xlabel("Признак 1 после StandardScaler")
    axis.set_ylabel("Признак 2 после StandardScaler")
    axis.legend(fontsize=8)

    return accuracy, len(support)


# 3. Линейное и RBF-ядро
linear_model = SVC(kernel="linear", C=1)
rbf_model = SVC(kernel="rbf", C=1, gamma="scale")

linear_model.fit(X_scaled, y)
rbf_model.fit(X_scaled, y)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

linear_accuracy, linear_support = plot_boundary(
    linear_model,
    X_scaled,
    y,
    "Линейное ядро, C=1",
    "svm_linear.png",
    axes[0]
)

rbf_accuracy, rbf_support = plot_boundary(
    rbf_model,
    X_scaled,
    y,
    "RBF-ядро, C=1",
    "svm_rbf.png",
    axes[1]
)

plt.tight_layout()
plt.savefig("kernel_comparison.png", dpi=180)
plt.close()


# 4. Исследование параметра C
c_values = [0.01, 1, 100]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

c_results = []

for axis, c_value in zip(axes, c_values):
    model = SVC(
        kernel="rbf",
        C=c_value,
        gamma="scale"
    )
    model.fit(X_scaled, y)

    accuracy, support_count = plot_boundary(
        model,
        X_scaled,
        y,
        f"RBF, C={c_value}",
        "",
        axis
    )

    c_results.append((c_value, accuracy, support_count))

plt.tight_layout()
plt.savefig("c_comparison.png", dpi=180)
plt.close()


# 5. Вывод результатов
print("\n=== Сравнение ядер ===")
print(
    f"Линейное ядро: accuracy={linear_accuracy:.3f}, "
    f"опорных векторов={linear_support}"
)
print(
    f"RBF-ядро:      accuracy={rbf_accuracy:.3f}, "
    f"опорных векторов={rbf_support}"
)

print("\n=== Влияние параметра C ===")
print(f"{'C':>8} {'Accuracy':>12} {'Опорных векторов':>20}")

for c_value, accuracy, support_count in c_results:
    print(f"{c_value:>8} {accuracy:>12.3f} {support_count:>20}")

print("\nВывод:")
print(
    "RBF лучше разделяет полумесяцы, потому что строит "
    "нелинейную изогнутую границу."
)
print(
    "Малое C допускает больше ошибок и создаёт более простую границу."
)
print(
    "Большое C сильнее наказывает ошибки и точнее подстраивается "
    "под обучающие точки, но повышает риск переобучения."
)

print("\nСозданы файлы:")
print("kernel_comparison.png")
print("c_comparison.png")
