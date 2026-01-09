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

def nested_lasso_cv(X, y, n_outer=5, n_inner=5, random_state=0):

    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=2026)

    y_pred = np.zeros_like(y, dtype=float)
    coef_all_folds = []
    rmse_all_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=20000))
        ])

        param_grid = {
            "model__alpha": np.logspace(-6, -1.1, 50)
        }

        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        coef = best_model.named_steps["model"].coef_
        coef_all_folds.append(coef)

        y_pred[test_idx] = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred[test_idx]))
        rmse_all_folds.append(rmse)

        print(
            f"Fold {fold_idx+1}: "
            f"best alpha = {gs.best_params_['model__alpha']:.2e}, "
            f"non-zero coef = {np.sum(coef != 0)}"
        )

    r, p = pearsonr(y, y_pred)

    return {
        "y_pred": y_pred,
        "r": r,
        "p": p,
        "rmse": np.mean(rmse_all_folds),
        "coef": np.array(coef_all_folds)
    }


rawDataCue01_Theta = sio.loadmat(r"D:\ZJF_Conn\NBSanalysis170\MLData\Cue01SThetaregress.mat")
X_Cue01_Theta = rawDataCue01_Theta["Thetaconn"]
y_Cue01_Theta = rawDataCue01_Theta["Fear"].ravel()[0:170]

rawDataCue01_Alpha = sio.loadmat(r'D:\ZJF_Conn\NBSanalysis170\MLData\Cue01SAlpharegress.mat') 
X_Cue01_Alpha = rawDataCue01_Alpha["Alphaconn"]
y_Cue01_Alpha = rawDataCue01_Alpha["Anti"].ravel()[0:170]

rawDataCue25_Alpha = sio.loadmat(r"D:\ZJF_Conn\NBSanalysis170\MLData\Cue25SAlpharegress.mat")
X_Cue25_Alpha = rawDataCue25_Alpha["Alphaconn"]
y_Cue25_Alpha = rawDataCue25_Alpha["Pain"].ravel()[0:170]

rawDataPain01_Beta = sio.loadmat(r'D:\ZJF_Conn\NBSanalysis170\MLData\Pain01SBetaregress.mat')
X_Pain01_Beta = rawDataPain01_Beta["Betaconn"]
y_Pain01_Beta = rawDataPain01_Beta["Pain"].ravel()[0:170]

for i in range(60):
    print(f"\n===== Repeat {i+1}/{60} =====")


    res_Cue01_Theta = nested_lasso_cv(X_Cue01_Theta, y_Cue01_Theta , random_state=i)
    plot_ytrue_ypred_joint(y_true=y_Cue01_Theta, y_pred=res_Cue01_Theta["y_pred"],
                            title=f"Lasso: True vs Predicted Cue01 Theta"+ str(i), r=res_Cue01_Theta['r'], p=res_Cue01_Theta['p'])

    res_Cue01_Alpha = nested_lasso_cv(X_Cue01_Alpha, y_Cue01_Alpha , random_state=i)
    plot_ytrue_ypred_joint(y_true=y_Cue01_Alpha, y_pred=res_Cue01_Alpha["y_pred"],
                            title=f"Lasso: True vs Predicted Cue01 Alpha"+ str(i), r=res_Cue01_Alpha['r'], p=res_Cue01_Alpha['p'])
    
    res_Cue25_Alpha = nested_lasso_cv(X_Cue25_Alpha, y_Cue25_Alpha , random_state=i)
    plot_ytrue_ypred_joint(y_true=y_Cue25_Alpha, y_pred=res_Cue25_Alpha["y_pred"],
                            title=f"Lasso: True vs Predicted Cue25 Alpha"+ str(i), r=res_Cue25_Alpha['r'], p=res_Cue25_Alpha['p'])

    res_Pain01_Beta = nested_lasso_cv(X_Pain01_Beta, y_Pain01_Beta , random_state=i)
    plot_ytrue_ypred_joint(y_true=y_Pain01_Beta, y_pred=res_Pain01_Beta["y_pred"],
                            title=f"Lasso: True vs Predicted Pain01 Alpha"+ str(i), r=res_Pain01_Beta['r'], p=res_Pain01_Beta['p'])

