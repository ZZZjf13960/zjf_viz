import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting

coord_file = r"D:\ZJF_Conn\NBSanalysis\MNI\Schaefer_MNI_sorted.txt"
coord_df = pd.read_csv(
    coord_file,
    sep=r"\s+",
    header=None,
    names=[ "x", "y", "z"]
)
node_coords = coord_df[['x', 'y', 'z']].values
###
adjCue01_anti_alpha = sio.loadmat(r"D:\ZJF_Conn\CodeML_regress\MLData\adjacency_Anti_AlphaCue01.mat")
adjCue01_anti_alpha = adjCue01_anti_alpha['adjacency']

plotting.plot_connectome(
    np.abs(adjCue01_anti_alpha),
    node_coords,
    edge_threshold="0%",  
    edge_vmin=np.min(np.abs(adjCue01_anti_alpha)) ,   
    edge_vmax=np.max(np.abs(adjCue01_anti_alpha)), 
    node_size=10,
    title='LASSO Weight (Cue01 Anti, Alpha)'
)
###
adjCue25_pain_alpha = sio.loadmat(r'D:\ZJF_Conn\CodeML_regress\MLData\adjacency_Pain_AlphaCue25.mat')
adjCue25_pain_alpha = adjCue25_pain_alpha['adjacency']

plotting.plot_connectome(
    np.abs(adjCue25_pain_alpha),
    node_coords,
    edge_threshold="0%",    
    edge_vmin=np.min(np.abs(adjCue25_pain_alpha)) ,   
    edge_vmax=np.max(np.abs(adjCue25_pain_alpha)), 
    node_size=10,
    title='LASSO Weight (Cue25 Pain, Alpha)'
)
###
adjCue25_fear_alpha = sio.loadmat(r'D:\ZJF_Conn\CodeML_regress\MLData\adjacency_Fear_AlphaCue25.mat')
adjCue25_fear_alpha = adjCue25_fear_alpha['adjacency']

plotting.plot_connectome(
    np.abs(adjCue25_fear_alpha),
    node_coords,
    edge_threshold="0%",    
    edge_vmin=np.min(np.abs(adjCue25_fear_alpha)) ,   
    edge_vmax=np.max(np.abs(adjCue25_fear_alpha)), 
    node_size=10,
    title='LASSO Weight (Cue25 Fear, Alpha)'
)
###
adjCue01_pain_theta170 = sio.loadmat(r'D:\ZJF_Conn\NBSanalysis170\MLData\adjacency_Pain_ThetaCue01.mat')
adjCue01_pain_theta170 = adjCue01_pain_theta170['adjacency']

plotting.plot_connectome(
    np.abs(adjCue01_pain_theta170),
    node_coords,
    edge_threshold="0%",    
    edge_vmin=np.min(np.abs(adjCue01_pain_theta170)) ,   
    edge_vmax=np.max(np.abs(adjCue01_pain_theta170)), 
    node_size=10,
    title='LASSO Weight (Cue01 Pain, Theta)'
)
###
adjCue01_anti_alpha170 = sio.loadmat(r'D:\ZJF_Conn\NBSanalysis170\MLData\adjacency_Anti_AlphaCue01.mat')
adjCue01_anti_alpha170 = adjCue01_anti_alpha170['adjacency']

plotting.plot_connectome(
    np.abs(adjCue01_anti_alpha170),
    node_coords,
    edge_threshold="0%",    
    edge_vmin=np.min(np.abs(adjCue01_anti_alpha170)) ,   
    edge_vmax=np.max(np.abs(adjCue01_anti_alpha170)), 
    node_size=10,
    title='LASSO Weight (Cue01 Anti, Alpha)'
)
###
adjCue25_pain_alpha170 = sio.loadmat(r'D:\ZJF_Conn\NBSanalysis170\MLData\adjacency_Pain_AlphaCue25.mat')
adjCue25_pain_alpha170 = adjCue25_pain_alpha170['adjacency']

plotting.plot_connectome(
    np.abs(adjCue25_pain_alpha170),
    node_coords,
    edge_threshold="0%",    
    edge_vmin=np.min(np.abs(adjCue25_pain_alpha170)) ,   
    edge_vmax=np.max(np.abs(adjCue25_pain_alpha170)), 
    node_size=10,
    title='LASSO Weight (Cue01 Anxiety, Alpha)'
)
###
adjPain01_unpleasant_beta170 = sio.loadmat(r'D:\ZJF_Conn\NBSanalysis170\MLData\adjacency_Unpleasant_BetaPain.mat')
adjPain01_unpleasant_beta170 = adjPain01_unpleasant_beta170['adjacency']

plotting.plot_connectome(
    np.abs(adjPain01_unpleasant_beta170),
    node_coords,
    edge_threshold="0%",    
    edge_vmin=np.min(np.abs(adjPain01_unpleasant_beta170)) ,   
    edge_vmax=np.max(np.abs(adjPain01_unpleasant_beta170)), 
    node_size=10,
    title='LASSO Weight (Pain01 Unpleasant, Beta)'
)
###
adjCue01_fear_theta = sio.loadmat(r'D:\ZJF_Conn\CodeML_regress\MLData\adjacency_Fear_ThetaCue01_9.mat')
adjCue01_fear_theta = adjCue01_fear_theta['adjacency']

plotting.plot_connectome(
    np.abs(adjCue01_fear_theta),
    node_coords,
    edge_threshold="0%",    
    edge_vmin=np.min(np.abs(adjCue01_fear_theta)) ,   
    edge_vmax=np.max(np.abs(adjCue01_fear_theta)), 
    node_size=10,
    title='LASSO Weight (Cue01 Fear, Beta)'
)
###
adjPain01_Pain_beta170 = sio.loadmat(r'D:\ZJF_Conn\NBSanalysis170\MLData\adjacency_Pain_BetaPain0155.mat')
adjPain01_Pain_beta170 = adjPain01_Pain_beta170['adjacency']

plotting.plot_connectome(
    np.abs(adjPain01_Pain_beta170),
    node_coords,
    edge_threshold="0%",    
    edge_vmin=np.min(np.abs(adjPain01_Pain_beta170)) ,   
    edge_vmax=np.max(np.abs(adjPain01_Pain_beta170)), 
    node_size=10,
    title='LASSO Weight (Pain01 Pain, Beta)'
)
