from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from common import load_split

X_train, X_test, y_train, y_test = load_split()
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

for thr in (0.5, 0.3, 0.1):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    p, r = tp / (tp + fp), tp / (tp + fn)
    print(f"порог {thr}: TP={tp} FN={fn} FP={fp} TN={tn} P={p:.3f} R={r:.3f}")
