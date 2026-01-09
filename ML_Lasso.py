import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns



sns.set(style="whitegrid")  

def plot_ytrue_ypred_joint(y_true, y_pred, title, r=None, p=None, color="#4c72b0"):
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

### 196 ： 29 43 
### 170 ： 34
rawData = sio.loadmat(r"D:\ZJF_Conn\NBSanalysis170\MLData\Pain01SBetaregress.mat")
X = rawData["Betaconn"]
y = rawData["Pain"].ravel()[0:170]
x_ = np.mean(X, axis=1)

# BehaveData = sio.loadmat(r"D:\ZJF_Conn\CodeML_regress\MLData\Behave.mat")
# y = BehaveData['Pain'].ravel()

cor, p = pearsonr(x_, y)
print(f"Mean feature correlation with y: r = {cor:.3f}, p = {p:.3g}")
print("X shape:", X.shape, "y shape:", y.shape)


outer_cv = KFold(n_splits=5, shuffle=True, random_state=55)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

y_pred = np.zeros_like(y)
coef_all_folds = []
rmse_all_folds = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=20000))
    ])

    param_grid = {"model__alpha": np.logspace(-6, -1.1, 50)}
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv,
                      scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)

    coef = gs.best_estimator_.named_steps['model'].coef_
    coef_all_folds.append(coef)

    y_pred[test_idx] = gs.predict(X_test)

    # ===== RMSE =====
    rmse = np.sqrt(mean_squared_error(y_test, y_pred[test_idx]))
    rmse_all_folds.append(rmse)

    print(f"Fold {fold_idx+1}: best alpha = {gs.best_params_['model__alpha']}, non-zero coef = {np.sum(coef!=0)}")


r, p = pearsonr(y, y_pred)
print(f"Cross-validated r = {r:.3f}, p = {p:.3g}")

plot_ytrue_ypred_joint(y_true=y, y_pred=y_pred,
                        title=f"Lasso: True vs Predicted", r=r, p=p)

coef_all_folds = np.array(coef_all_folds)
mean_coef = coef_all_folds.mean(axis=0)
nonzero_count = (coef_all_folds != 0).sum(axis=0)

nonzero_idx = np.where(nonzero_count > 0)[0]

plt.figure(figsize=(10,4))
plt.bar(nonzero_idx, mean_coef[nonzero_idx], color='skyblue')
plt.xlabel("Feature index")
plt.ylabel("Average coefficient")
plt.title(f"Lasso coefficients across {outer_cv.n_splits} folds (non-zero in ≥1 fold)")

plt.figure(figsize=(10,4))
plt.bar(nonzero_idx, nonzero_count[nonzero_idx], color='salmon')
plt.xlabel("Feature index")
plt.ylabel("Non-zero frequency across folds")
plt.title("Number of folds where coefficient is non-zero")

# save_path = r"D:\ZJF_Conn\CodeML_regress\MLData\Pain01SLasso_coefficients_Pain_Beta55.mat"

# sio.savemat(save_path, {
#     "mean_coef": mean_coef,
#     "coef_all_folds": coef_all_folds
# })
