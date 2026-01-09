import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

def get_network(roi_name):
    for net in networks:
        if net in roi_name:
            return net
    return 'Unknown'

rawData = sio.loadmat(r"D:\ZJF_Conn\NBSanalysis170\MLData\Pain01SLasso_coefficients_Unpleasant_Alpha.mat")

mean_coef = rawData["mean_coef"]
coef_all_folds = rawData["coef_all_folds"]
print("mean_coef shape:", mean_coef.shape, "coef_all_folds shape:", coef_all_folds.shape)

roiLabel = sio.loadmat(r"D:\ZJF_Conn\NBSanalysis170\MLData\Roiname_Pain01Beta.mat")
roiLabel = roiLabel['Roiname']
roi_pairs = [s[0][0].split('-') for s in roiLabel]

df = pd.DataFrame({
    'ROI1': [p[0] for p in roi_pairs],
    'ROI2': [p[1] for p in roi_pairs],
    'mean_coef': mean_coef.flatten()
})

networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Cont', 'Limbic', 'Default']
df['Net1'] = df['ROI1'].apply(get_network)
df['Net2'] = df['ROI2'].apply(get_network)

network_values = {net: [] for net in networks}


for i, row in df.iterrows():
    coef_abs = abs(row['mean_coef'])
    network_values[row['Net1']].append(coef_abs)
    network_values[row['Net2']].append(coef_abs)

network_importance = {net: np.mean(vals) if len(vals)>0 else 0 
                      for net, vals in network_values.items()}

net_df = pd.DataFrame({
    'Network': list(network_importance.keys()),
    'Importance': list(network_importance.values())
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
plt.bar(net_df['Network'], net_df['Importance'], color='skyblue')
plt.ylabel('Mean abs(coef) per node')
plt.title('Normalized Network importance (LASSO coefficients)')

Net_pairs = [
    [get_network(p[0]), get_network(p[1])]
    for p in roi_pairs
]
df['FeatureName'] = Net_pairs

df_nonzero = df[df['mean_coef'] != 0].copy()
df_nonzero = df_nonzero.reindex(df_nonzero['mean_coef'].abs().sort_values(ascending=False).index)

plt.figure(figsize=(12,6))
colors = ['red' if v>0 else 'blue' for v in df_nonzero['mean_coef']]
plt.barh(range(len(df_nonzero)), df_nonzero['mean_coef'], color=colors)
plt.yticks(range(len(df_nonzero)), df_nonzero['FeatureName'])  
plt.xlabel('LASSO coefficient (mean_coef)')
plt.title(f'All non-zero connections ({len(df_nonzero)} edges)')
plt.gca().invert_yaxis()
plt.tight_layout()


network_values = {net: [] for net in networks}

for i, row in df_nonzero.iterrows():
    coef_abs = abs(row['mean_coef'])
    network_values[row['Net1']].append(coef_abs)
    network_values[row['Net2']].append(coef_abs)
    
network_importance = {net: np.mean(vals) if len(vals) > 0 else 0 
                      for net, vals in network_values.items()}

net_df = pd.DataFrame({
    'Network': list(network_importance.keys()),
    'Importance': list(network_importance.values())
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
plt.bar(net_df['Network'], net_df['Importance'], color='skyblue')
plt.ylabel('Mean abs(coef) per node (non-zero only)')
plt.title('Normalized Network importance (LASSO coefficients, non-zero)')
