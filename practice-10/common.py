from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_split():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    y = 1 - y  # теперь 1 = злокачественная опухоль
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
