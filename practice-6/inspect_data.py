import pandas as pd

data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    names=["label", "text"],
)

print("Размер датасета:", data.shape)

print("\nКоличество сообщений по классам:")
print(data["label"].value_counts())

print("\nДоли классов:")
print(data["label"].value_counts(normalize=True).round(3))

print("\nПервые сообщения:")
print(data.head())
