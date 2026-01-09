import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.io as sio

sns.set(style="whitegrid")  


rawData = sio.loadmat(r"D:\ZJF_Conn\NBSanalysis170\MLData\Pain01SBetaregress.mat")
rawData196 = sio.loadmat(r"D:\ZJF_Conn\Data\MLData\Cue25SAlpharegress.mat")
X = rawData196["Alphaconn"]
Y = np.concatenate([rawData['Pain'],rawData['Anti'].T,rawData['Unpleasant'],rawData['Fear'].T,rawData['Anxiety'].T],axis = 1)
Y = Y

for i in range(60):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=44)

    y_pred = np.zeros_like(Y)
    weights_all_folds = [] 
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", PLSRegression())
        ])
        param_grid = {'model__n_components': np.arange(1, min(X.shape[1], 10)+1)} 
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv,
                        scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(X_train, y_train)
        print("Best n_components:", gs.best_params_)
        print("Best CV MSE:", -gs.best_score_)
        pls_model = gs.best_estimator_.named_steps['model']
        weights_all_folds.append(pls_model.x_weights_)

        y_pred[test_idx] = gs.predict(X_test)


    output_names = ['Pain', 'Anti', 'Unpleasant', 'Fear', 'Anxiety']

    df_true = pd.DataFrame(Y, columns=output_names)
    df_pred = pd.DataFrame(y_pred, columns=output_names)
    corr_matrix = np.corrcoef(df_pred.T, df_true.T)[:5, 5:] 

    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, xticklabels=output_names, yticklabels=output_names, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel("True Y")
    plt.ylabel("Predicted Y")
    plt.title("Correlation: Predicted vs True Y")





