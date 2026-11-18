import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.datasets import make_moons
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
DBSCAN_EPS = 0.25
MIN_SAMPLES = 5


# --------------------------------------------------
# 1. Данные сложной невыпуклой формы
# --------------------------------------------------

X, hidden_labels = make_moons(
    n_samples=500,
    noise=0.07,
    random_state=RANDOM_STATE,
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Данные сложной формы ===")
print("Объектов:", len(X_scaled))
print("Признаков:", X_scaled.shape[1])
print("Истинные метки скрыты от алгоритмов.")


# --------------------------------------------------
# 2. График k-расстояний для DBSCAN
# --------------------------------------------------

neighbors = NearestNeighbors(
    n_neighbors=MIN_SAMPLES,
)

neighbors.fit(X_scaled)

distances, _ = neighbors.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 5))
plt.plot(k_distances, color="royalblue")
plt.axhline(
    DBSCAN_EPS,
    color="red",
    linestyle="--",
    label=f"Выбранный eps={DBSCAN_EPS}",
)
plt.xlabel("Объекты после сортировки")
plt.ylabel(f"Расстояние до {MIN_SAMPLES}-го соседа")
plt.title("График k-расстояний для выбора eps")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("k_distance.png", dpi=160)
plt.close()


# --------------------------------------------------
# 3. Три алгоритма кластеризации
# --------------------------------------------------

models = {
    "K-Means": KMeans(
        n_clusters=2,
        n_init=20,
        random_state=RANDOM_STATE,
    ),
    "DBSCAN": DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=MIN_SAMPLES,
    ),
    "Иерархический": AgglomerativeClustering(
        n_clusters=2,
        linkage="single",
    ),
}

results = {}
labels_by_model = {}

print("\n=== Сравнение методов ===")
print(
    f"{'Метод':<18}"
    f"{'Кластеров':>11}"
    f"{'Шум':>8}"
    f"{'Силуэт':>12}"
    f"{'ARI':>10}"
)

for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    labels_by_model[name] = labels

    non_noise = labels != -1
    clean_labels = labels[non_noise]
    clean_data = X_scaled[non_noise]

    unique_clusters = set(clean_labels)
    n_clusters = len(unique_clusters)
    noise_count = int((labels == -1).sum())

    if n_clusters > 1 and len(clean_data) > n_clusters:
        silhouette = silhouette_score(
            clean_data,
            clean_labels,
        )
    else:
        silhouette = float("nan")

    ari = adjusted_rand_score(
        hidden_labels,
        labels,
    )

    results[name] = {
        "clusters": n_clusters,
        "noise": noise_count,
        "silhouette": silhouette,
        "ari": ari,
    }

    print(
        f"{name:<18}"
        f"{n_clusters:>11}"
        f"{noise_count:>8}"
        f"{silhouette:>12.3f}"
        f"{ari:>10.3f}"
    )


# --------------------------------------------------
# 4. Визуальное сравнение
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.5),
)

for axis, (name, labels) in zip(
    axes,
    labels_by_model.items(),
):
    axis.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        cmap="viridis",
        s=22,
        edgecolor="black",
        linewidth=0.15,
    )

    score = results[name]["ari"]

    axis.set_title(
        f"{name}\nARI={score:.3f}"
    )
    axis.set_xlabel("Признак 1")
    axis.set_ylabel("Признак 2")
    axis.grid(alpha=0.2)

plt.suptitle(
    "Кластеризация данных невыпуклой формы",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(
    "methods_comparison.png",
    dpi=160,
)
plt.close()


# --------------------------------------------------
# 5. Дендрограмма
# --------------------------------------------------

linkage_matrix = linkage(
    X_scaled,
    method="single",
)

plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    show_contracted=True,
)

plt.title("Дендрограмма иерархической кластеризации")
plt.xlabel("Группы объектов")
plt.ylabel("Расстояние объединения")
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=160)
plt.close()


# --------------------------------------------------
# 6. Вывод
# --------------------------------------------------

best_method = max(
    results,
    key=lambda name: results[name]["ari"],
)

print("\n=== Вывод ===")
print("Лучший метод по ARI:", best_method)

print(
    "K-Means строит округлые кластеры и плохо "
    "повторяет форму полумесяцев."
)

print(
    "DBSCAN находит кластеры произвольной формы "
    "и может выделять шум."
)

print(
    "Иерархическая кластеризация с single linkage "
    "также способна восстановить форму полумесяцев."
)

print("\n=== Созданные файлы ===")
print("k_distance.png")
print("methods_comparison.png")
print("dendrogram.png")
