import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


# --------------------------------------------------
# 1. Создание учебных данных клиентов
# --------------------------------------------------

centers = np.array([
    [-1.2, -1.0, -0.8],
    [0.2, 1.2, 0.1],
    [1.4, -0.1, 1.4],
])

latent, _ = make_blobs(
    n_samples=360,
    centers=centers,
    cluster_std=0.45,
    random_state=RANDOM_STATE,
)

df = pd.DataFrame({
    "annual_spend": np.clip(latent[:, 0] * 250 + 600, 50, None),
    "visits_per_month": np.clip(latent[:, 1] * 2.5 + 6, 1, None),
    "avg_check": np.clip(latent[:, 2] * 50 + 120, 10, None),
})

print("=== Данные ===")
print("Количество клиентов:", len(df))
print("Пропусков:", int(df.isna().sum().sum()))
print(df.describe().round(2))


# --------------------------------------------------
# 2. Масштабирование
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print("\nСредние после масштабирования:")
print(np.round(X_scaled.mean(axis=0), 3))

print("Стандартные отклонения:")
print(np.round(X_scaled.std(axis=0), 3))


# --------------------------------------------------
# 3. Метод локтя и коэффициент силуэта
# --------------------------------------------------

k_values = list(range(2, 11))
inertias = []
silhouettes = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=RANDOM_STATE,
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))


best_index = int(np.argmax(silhouettes))
best_k = k_values[best_index]

print("\n=== Выбор количества кластеров ===")
print(f"{'k':>3} {'Инерция':>12} {'Силуэт':>12}")

for k, inertia, silhouette in zip(
    k_values,
    inertias,
    silhouettes,
):
    print(f"{k:>3} {inertia:>12.2f} {silhouette:>12.3f}")

print("\nЛучшее k по силуэту:", best_k)


# --------------------------------------------------
# 4. График локтя и силуэта
# --------------------------------------------------

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(
    k_values,
    inertias,
    marker="o",
    color="royalblue",
    label="Инерция",
)
ax1.set_xlabel("Количество кластеров k")
ax1.set_ylabel("Инерция", color="royalblue")
ax1.tick_params(axis="y", labelcolor="royalblue")
ax1.set_xticks(k_values)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(
    k_values,
    silhouettes,
    marker="s",
    color="darkorange",
    label="Силуэт",
)
ax2.set_ylabel("Silhouette score", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

plt.title("Метод локтя и коэффициент силуэта")
plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=160)
plt.close()


# --------------------------------------------------
# 5. Итоговая модель
# --------------------------------------------------

final_model = KMeans(
    n_clusters=best_k,
    n_init=20,
    random_state=RANDOM_STATE,
)

final_labels = final_model.fit_predict(X_scaled)

result = df.copy()
result["cluster"] = final_labels

profiles = (
    result.groupby("cluster")
    .mean()
    .round(2)
)

# Названия определяем по расходам клиентов
ordered_clusters = (
    profiles["annual_spend"]
    .sort_values()
    .index
    .tolist()
)

segment_names = {}

for rank, cluster in enumerate(ordered_clusters):
    if rank == 0:
        name = "Экономные клиенты"
    elif rank == len(ordered_clusters) - 1:
        name = "Премиальные клиенты"
    else:
        name = "Регулярные клиенты"

    segment_names[cluster] = name

profiles["segment"] = [
    segment_names[cluster]
    for cluster in profiles.index
]

print("\n=== Профили кластеров ===")
print(profiles.to_string())


# --------------------------------------------------
# 6. Визуализация кластеров
# --------------------------------------------------

centers_original = scaler.inverse_transform(
    final_model.cluster_centers_
)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    df["annual_spend"],
    df["visits_per_month"],
    c=final_labels,
    cmap="viridis",
    alpha=0.7,
    edgecolor="black",
    linewidth=0.2,
)

plt.scatter(
    centers_original[:, 0],
    centers_original[:, 1],
    marker="X",
    s=250,
    c="red",
    edgecolor="black",
    label="Центры",
)

plt.xlabel("Годовые расходы")
plt.ylabel("Посещений в месяц")
plt.title(f"Сегментация клиентов: K-Means, k={best_k}")
plt.legend()
plt.colorbar(scatter, label="Номер кластера")
plt.tight_layout()
plt.savefig("cluster_profiles.png", dpi=160)
plt.close()


# --------------------------------------------------
# 7. Проверка устойчивости
# --------------------------------------------------

second_model = KMeans(
    n_clusters=best_k,
    n_init=20,
    random_state=7,
)

second_labels = second_model.fit_predict(X_scaled)

stability = adjusted_rand_score(
    final_labels,
    second_labels,
)

print("\n=== Устойчивость ===")
print("Adjusted Rand Index:", round(stability, 3))

if stability >= 0.9:
    print("Состав кластеров практически не изменился.")
elif stability >= 0.7:
    print("Результат достаточно устойчив.")
else:
    print("Результат заметно зависит от инициализации.")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    df["annual_spend"],
    df["visits_per_month"],
    c=final_labels,
    cmap="viridis",
    s=25,
)
axes[0].set_title("random_state=42")
axes[0].set_xlabel("Годовые расходы")
axes[0].set_ylabel("Посещений в месяц")

axes[1].scatter(
    df["annual_spend"],
    df["visits_per_month"],
    c=second_labels,
    cmap="viridis",
    s=25,
)
axes[1].set_title("random_state=7")
axes[1].set_xlabel("Годовые расходы")
axes[1].set_ylabel("Посещений в месяц")

plt.suptitle(f"Проверка устойчивости, ARI={stability:.3f}")
plt.tight_layout()
plt.savefig("stability_comparison.png", dpi=160)
plt.close()


print("\n=== Созданные файлы ===")
print("elbow_silhouette.png")
print("cluster_profiles.png")
print("stability_comparison.png")
