# Проверка преподавателя: 5-fold кросс-валидация тех же моделей (уровень C)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
y = 1 - y
models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Tree": DecisionTreeClassifier(random_state=42),
    "Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Boosting": GradientBoostingClassifier(random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    s = cross_val_score(model, X, y, cv=cv)
    print(f"{name:<9} {s.mean():.3f} ± {s.std():.3f}")
