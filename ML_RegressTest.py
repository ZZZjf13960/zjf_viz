import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV, BayesianRidge
from sklearn.kernel_ridge import KernelRidge


rawData = sio.loadmat(r"D:\ZJF_Conn\Data\MLData\Cue25SAlpharegress.mat")
X = rawData["Alphaconn"]
y = rawData["PainDiff"].ravel()

print("X shape:", X.shape, "y shape:", y.shape)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)


models = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5))
    ]),

    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=np.linspace(0.05, 1.0, 25),
            alphas=np.logspace(-4, 2, 40),
            cv=5,
            max_iter=20000,
            n_jobs=-1
        ))
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(
            alphas=np.logspace(-4, 1, 50),
            cv=5,
            max_iter=20000,
            n_jobs=-1
        ))
    ]),

    "BayesianRidge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", BayesianRidge())
    ]),

    "KernelRidge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KernelRidge(
            kernel="rbf",
            alpha=1.0,
            gamma=1e-3
        ))
    ])
}



results = {}

for name, pipe in models.items():
    print(f"\nRunning model: {name}")

    y_pred = cross_val_predict(
        pipe,
        X,
        y,
        cv=outer_cv,
        n_jobs=10
    )

    r, p = pearsonr(y, y_pred)

    results[name] = {
        "r": r,
        "p": p,
        "y_pred": y_pred
    }
    
    print(f"{name}: r = {r:.4f}, p = {p:.4g}")

print("\n========== Summary ==========")
for k, v in results.items():
    print(f"{k:15s}  r = {v['r']:.4f}   p = {v['p']:.4g}")
