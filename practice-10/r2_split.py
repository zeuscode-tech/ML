from common import load_split

X_train, X_test, y_train, y_test = load_split()
print("train:", X_train.shape, "test:", X_test.shape)
print("Доля злокачественных в train:", round(y_train.mean(), 3))
print("Доля злокачественных в test: ", round(y_test.mean(), 3))
