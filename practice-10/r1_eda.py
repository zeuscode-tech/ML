from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame
print("Размер:", df.shape)
print("Пропусков:", df.isna().sum().sum())
print("Классы:", df["target"].value_counts().to_dict())
print("0 =", data.target_names[0], "| 1 =", data.target_names[1])
print(df[["mean radius", "mean area"]].describe().round(1))
