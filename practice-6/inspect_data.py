import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent

data = pd.read_csv(
    HERE / "data" / "SMSSpamCollection",
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
