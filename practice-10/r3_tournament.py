import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score
from common import load_split

X_train, X_test, y_train, y_test = load_split()
models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Tree": DecisionTreeClassifier(random_state=42),
    "Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Boosting": GradientBoostingClassifier(random_state=42),
}

print(f"{'модель':<9}{'acc_train':>10}{'acc_test':>10}{'recall':>8}{'сек':>7}")
for name, model in models.items():
    t = time.perf_counter()
    model.fit(X_train, y_train)
    sec = time.perf_counter() - t
    acc_tr = accuracy_score(y_train, model.predict(X_train))
    pred = model.predict(X_test)
    acc_te = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred)
    print(f"{name:<9}{acc_tr:>10.3f}{acc_te:>10.3f}{rec:>8.3f}{sec:>7.2f}")
