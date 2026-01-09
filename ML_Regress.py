import os
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV,LassoCV
from tqdm import tqdm


rawData = sio.loadmat(r"D:\ZJF_Conn\Data\MLData\Cue01SThetaregress.mat")
X= rawData['Thetaconn']
y = rawData['AntiDiff'].ravel()
print('X的shape:',X.shape,'y的shape:',y.shape)


l1_ratio_list = np.linspace(0.05, 1.0, 100)   
alphas = np.logspace(-4, 2, 50)


enet = ElasticNetCV(
    l1_ratio=l1_ratio_list,
    alphas=alphas,
    cv=5,                
    max_iter=10000,
    n_jobs=-1
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", enet)
])

outer_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


y_pred = cross_val_predict(
    pipe,
    X,
    y,
    cv=outer_cv,
    n_jobs=10
)

r, p = pearsonr(y, y_pred)

print("=" * 60)
print("Cross-validated performance")
print(f"Pearson r = {r:.4f}")
print(f"P-value   = {p:.4g}")
print("=" * 60)
