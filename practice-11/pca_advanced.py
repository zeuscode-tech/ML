import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler


# 1. Загружаем и масштабируем данные
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Данные ===")
print("Объектов:", X_scaled.shape[0])
print("Признаков:", X_scaled.shape[1])
print("Классы:", np.bincount(y))


def measure_embedding(name, model):
    """Строит двумерное представление и измеряет время."""
    start = time.perf_counter()
    embedding = model.fit_transform(X_scaled)
    elapsed = time.perf_counter() - start

    score = trustworthiness(
        X_scaled,
        embedding,
        n_neighbors=10,
    )

    print(
        f"{name:<12} "
        f"время={elapsed:.3f} с  "
        f"trustworthiness={score:.4f}"
    )

    return embedding, elapsed, score


# 2. PCA
pca_embedding, pca_time, pca_score = measure_embedding(
    "PCA",
    PCA(n_components=2),
)

# 3. t-SNE
tsne_model = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    max_iter=1000,
    random_state=42,
)

tsne_embedding, tsne_time, tsne_score = measure_embedding(
    "t-SNE",
    tsne_model,
)

# 4. UMAP
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
)

umap_embedding, umap_time, umap_score = measure_embedding(
    "UMAP",
    umap_model,
)


# 5. Сравнение трёх методов
embeddings = [
    ("PCA", pca_embedding),
    ("t-SNE", tsne_embedding),
    ("UMAP", umap_embedding),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for axis, (name, embedding) in zip(axes, embeddings):
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap="coolwarm",
        s=22,
        alpha=0.75,
        edgecolors="none",
    )

    axis.set_title(name)
    axis.set_xlabel("Компонента 1")
    axis.set_ylabel("Компонента 2")
    axis.grid(alpha=0.2)

handles, _ = scatter.legend_elements()
fig.legend(
    handles,
    ["malignant", "benign"],
    loc="upper center",
    ncol=2,
)

fig.suptitle(
    "Сравнение PCA, t-SNE и UMAP",
    fontsize=15,
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    "embedding_comparison.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()


# 6. Влияние perplexity на t-SNE
perplexities = [5, 30, 50]
tsne_results = []

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for axis, perplexity in zip(axes, perplexities):
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=42,
    )

    start = time.perf_counter()
    embedding = model.fit_transform(X_scaled)
    elapsed = time.perf_counter() - start

    score = trustworthiness(
        X_scaled,
        embedding,
        n_neighbors=10,
    )

    tsne_results.append({
        "perplexity": perplexity,
        "time_sec": elapsed,
        "trustworthiness": score,
    })

    axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap="coolwarm",
        s=22,
        alpha=0.75,
        edgecolors="none",
    )

    axis.set_title(
        f"perplexity={perplexity}\n"
        f"trust={score:.3f}"
    )
    axis.set_xlabel("Ось 1")
    axis.set_ylabel("Ось 2")
    axis.grid(alpha=0.2)

plt.suptitle("Влияние perplexity на результат t-SNE")
plt.tight_layout()
plt.savefig(
    "tsne_perplexity.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()


# 7. Можно ли преобразовывать новые объекты?
pca_supports_transform = hasattr(
    PCA(n_components=2),
    "transform",
)

tsne_supports_transform = hasattr(
    TSNE(n_components=2),
    "transform",
)

umap_supports_transform = hasattr(
    umap.UMAP(n_components=2),
    "transform",
)

print("\n=== Работа с новыми объектами ===")
print("PCA имеет transform:", pca_supports_transform)
print("t-SNE имеет transform:", tsne_supports_transform)
print("UMAP имеет transform:", umap_supports_transform)


# 8. Таблица результатов
comparison = pd.DataFrame([
    {
        "method": "PCA",
        "time_sec": pca_time,
        "trustworthiness": pca_score,
        "supports_transform": pca_supports_transform,
    },
    {
        "method": "t-SNE",
        "time_sec": tsne_time,
        "trustworthiness": tsne_score,
        "supports_transform": tsne_supports_transform,
    },
    {
        "method": "UMAP",
        "time_sec": umap_time,
        "trustworthiness": umap_score,
        "supports_transform": umap_supports_transform,
    },
])

comparison.to_csv(
    "embedding_results.csv",
    index=False,
)

pd.DataFrame(tsne_results).to_csv(
    "tsne_perplexity_results.csv",
    index=False,
)

print("\n=== Итоговое сравнение ===")
print(
    comparison.to_string(
        index=False,
        formatters={
            "time_sec": "{:.3f}".format,
            "trustworthiness": "{:.4f}".format,
        },
    )
)

print("\n=== t-SNE: влияние perplexity ===")
print(
    pd.DataFrame(tsne_results).to_string(
        index=False,
        formatters={
            "time_sec": "{:.3f}".format,
            "trustworthiness": "{:.4f}".format,
        },
    )
)

print("\n=== Вывод ===")
print(
    "PCA работает быстро, сохраняет глобальную структуру "
    "и может использоваться внутри ML-пайплайна."
)
print(
    "t-SNE хорошо показывает локальные группы, но не имеет "
    "корректного transform для новых объектов, поэтому обычно "
    "используется только для визуализации."
)
print(
    "UMAP сохраняет локальную структуру и поддерживает transform, "
    "поэтому его можно применять и к новым объектам."
)
print(
    "Результат t-SNE зависит от perplexity, поэтому выводы нельзя "
    "делать только по одному запуску."
)

print("\n=== Созданные файлы ===")
print("embedding_comparison.png")
print("tsne_perplexity.png")
print("embedding_results.csv")
print("tsne_perplexity_results.csv")
