import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  


def plot_ytrue_ypred_joint(y_true, y_pred, title, r=None, p=None, color="#4c72b0"):
    """
    绘制 y_true vs y_pred 的散点图 + 边际直方图 + 回归线 + CI
    只显示 r 和 p
    """
    g = sns.jointplot(
        x=y_true, 
        y=y_pred, 
        kind="reg",      
        scatter_kws={"s": 40, "alpha": 0.7, "color": color},
        line_kws={"color": "red"},
        ci=95,           
        marginal_kws={"bins": 15, "fill": True, "color": color}
    )

    g.fig.suptitle(title, fontsize=14)
    g.fig.subplots_adjust(top=0.92)

    if r is not None:
        g.ax_joint.text(
            0.05, 0.95,
            f"r = {r:.3f}\np = {p:.3g}",
            transform=g.ax_joint.transAxes,
            verticalalignment="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
    
    g.set_axis_labels("True y", "Predicted y", fontsize=12)



rawData = sio.loadmat(r"D:\ZJF_Conn\Data\MLData\Cue25SAlpharegress.mat")
X = rawData["Alphaconn"]
y = rawData["PainDiff"].ravel()

print("X shape:", X.shape, "y shape:", y.shape)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=43)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2026)


models = {
    "Ridge": (
        Ridge(),
        {"model__alpha": np.logspace(-3, 2, 50)}
    ),

    "Lasso": (
        Lasso(max_iter=20000),
        {"model__alpha": np.logspace(-6, -1.1, 50)}
    )
}

# =====================================================
# Run models
# =====================================================
results = {}
final_models = {}
for name, (model, param_grid) in models.items():

    print(f"\n========== {name} ==========")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    if param_grid:
        estimator = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        estimator.fit(X, y)
        best_model = estimator.best_estimator_

        print("Best params:", estimator.best_params_)

        final_models[name] = {
            "estimator": best_model,
            "best_params": estimator.best_params_
        }
    else:
        estimator = pipe
        best_model = pipe
        final_models[name] = {
            "estimator": best_model,
            "best_params": None
        }

    y_pred = cross_val_predict(
        estimator,
        X,
        y,
        cv=outer_cv,
        n_jobs=-1
    )

    # ---- Metrics ----
    r, p = pearsonr(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    results[name] = {
        "r": r,
        "p": p,
        "mse": mse,
        "r2": r2,
        "y_pred": y_pred
    }

    print(f"r = {r:.3f}, p = {p:.3g}, R² = {r2:.3f}, MSE = {mse:.3f}")

# =====================================================
# Summary
# =====================================================
print("\n========== Summary ==========")
for k, v in results.items():
    print(
        f"{k:15s} | "
        f"r = {v['r']:.3f} | "
        f"p = {v['p']:.3g} | "
        f"R² = {v['r2']:.3f}"
    )

for name, v in results.items():
    plot_ytrue_ypred_joint(
        y_true=y,
        y_pred=v["y_pred"],
        title=f"{name}: True vs Predicted",
        r=v["r"],
        p=v["p"]
    )

lasso_model = final_models['Lasso']['estimator']  
lasso_coef = lasso_model.named_steps['model'].coef_
print("Lasso coefficients shape:", lasso_coef.shape)
print(lasso_coef)
lasso_coef==0
nonzero_idx = np.where(lasso_coef != 0)[0]

plt.figure(figsize=(8, 4))
plt.bar(nonzero_idx, lasso_coef[nonzero_idx], color='skyblue')
plt.xlabel("Feature index")
plt.ylabel("Coefficient value")
plt.title("Non-zero Lasso coefficients")

