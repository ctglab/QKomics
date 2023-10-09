import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import glob
import pandas as pd
import os

from statistics import mean,variance

from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
import pickle
from utils import Split_and_sample, Kernel_concentration
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score




def Scale(x, out_range=(-1, 1), axis=None):
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


    

def calculate_intra_inter_distances(dist_matrix, labels):
    num_samples = len(labels)
    intra_cluster_distances = []
    inter_cluster_distances = []

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if labels[i] == labels[j]:
                intra_cluster_distances.append(dist_matrix[i, j])
            else:
                inter_cluster_distances.append(dist_matrix[i, j])

    return intra_cluster_distances, inter_cluster_distances

def Cluster_score(dist_matrix,labels):
    intra_cluster,inter_cluster=calculate_intra_inter_distances(dist_matrix,labels)
    m_intra=mean(intra_cluster)
    m_inter=mean(inter_cluster)
    #var
    v_intra=variance(intra_cluster)
    v_inter=variance(inter_cluster)
    #Score
    score=1-(m_intra/m_inter)
    return score, v_intra, v_inter




#############PLOTS###########################################

def Visual_comp(data_t,n_var,clusters,tag):
    #Plot first two components of given reduction algorithm and colors points according to given cluster
    col=[]
    for i in range(1,n_var+1):
        name='Component_'+str(i)
        col.append(name)
    pc_df = pd.DataFrame(data =data_t , columns =col ) 
    pc_df['Cluster'] =clusters
    print('########################weeeeee#############')
    print(pc_df.head())
    #plot pca
    g=sns.lmplot( x="Component_1", y="Component_2",
    data=pc_df, 
    palette=sns.color_palette("Paired"),
    fit_reg=False, 
    hue='Cluster', # color by cluster
    legend=True,
    scatter_kws={"s": 80}).set(title='First two components of '+tag)
  
    return g


def Silhouette_plot(X,K,scale=False,out_dir='./',tag=''):
    distortions = []

    X_dist=1-X
    if scale==True: X_dist=Scale(X_dist,(0,1))
    
   
    for k in K:
        spectral = SpectralClustering(k, affinity="precomputed",n_init=10)
        cluster_labels = spectral.fit_predict(X)
        #print(np.diagonal(q_k_dist))
        #set diagonal elements to 0
        np.fill_diagonal(X_dist,0)
        sil=silhouette_score(X_dist,metric='precomputed',labels=cluster_labels)
        distortions.append(sil)

    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette score showing the optimal k')
    plt.savefig(out_dir+ tag+'.png')
    plt.close()
    return 0