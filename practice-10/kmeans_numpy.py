import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs


# Данные создаём через sklearn,
# но сам алгоритм K-Means реализуем только на NumPy
X, _ = make_blobs(
    n_samples=150,
    centers=3,
    cluster_std=0.9,
    random_state=42,
)


def kmeans_np(X, start_idx, max_iter=100):
    centers = X[start_idx].copy()
    history = [centers.copy()]

    for iteration in range(1, max_iter + 1):
        # Расстояние от каждой точки до каждого центра
        distances = np.linalg.norm(
            X[:, None, :] - centers[None, :, :],
            axis=2,
        )

        # Номер ближайшего центра
        labels = distances.argmin(axis=1)

        new_centers = []

        for cluster in range(len(centers)):
            cluster_points = X[labels == cluster]

            # Защита от пустого кластера
            if len(cluster_points) == 0:
                new_center = centers[cluster]
            else:
                new_center = cluster_points.mean(axis=0)

            new_centers.append(new_center)

        new_centers = np.array(new_centers)
        history.append(new_centers.copy())

        # Центры перестали изменяться
        if np.allclose(
            new_centers,
            centers,
            atol=1e-6,
        ):
            centers = new_centers
            break

        centers = new_centers

    # Пересчитываем метки относительно финальных центров
    distances = np.linalg.norm(
        X[:, None, :] - centers[None, :, :],
        axis=2,
    )
    labels = distances.argmin(axis=1)

    # Сумма квадратов расстояний до своих центров
    sse = ((X - centers[labels]) ** 2).sum()

    return history, labels, iteration, sse


rng = np.random.default_rng(42)

start_idx = rng.choice(
    len(X),
    size=3,
    replace=False,
)

history, labels, iterations, sse = kmeans_np(
    X,
    start_idx,
)

history_array = np.array(history)
final_centers = history_array[-1]


print("=== K-Means на NumPy ===")
print("Объектов:", len(X))
print("Кластеров:", len(final_centers))
print("Начальные индексы:", start_idx.tolist())
print("Итераций:", iterations)
print("SSE:", round(sse, 3))
print("Финальные центры:")
print(np.round(final_centers, 3))


# Визуализация движения центров
plt.figure(figsize=(9, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7,
    edgecolor="black",
    linewidth=0.2,
)

for cluster in range(len(final_centers)):
    path = history_array[:, cluster, :]

    plt.plot(
        path[:, 0],
        path[:, 1],
        marker="o",
        linewidth=2,
        label=f"Путь центра {cluster}",
    )

plt.scatter(
    final_centers[:, 0],
    final_centers[:, 1],
    marker="X",
    s=250,
    c="red",
    edgecolor="black",
    label="Финальные центры",
)

plt.title(
    f"K-Means на NumPy: итераций={iterations}, "
    f"SSE={sse:.1f}"
)
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_history.png", dpi=160)
plt.close()


print("Создан файл: kmeans_history.png")
