import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import glob
import pandas as pd
import os

from statistics import mean,variance

from sklearn.cluster import SpectralClustering
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score, silhouette_samples




def Scale(x, out_range=(-1, 1), axis=None):
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2



def Get_centroids(X,label):
#Get centroids of N clusters################################
#X:array(N_samples,N_features)                             #
#label:array/list (N_samples) with cluster                 #
#returns array(N_cluster,N_feature) with centroids coord   #
############################################################
   label=np.asarray(label)
   classes=np.unique(label)
   centroids_=[]
   for i in range(0,len(classes)):
      c_=sum(X[tuple([label==classes[i]])])/len(label[tuple([label==classes[i]])])
      centroids_.append(c_)
   centroids=np.vstack(centroids_) 
   return centroids
    

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

def Calinski_Harabasz_Index(X,distance_matrix,labels):
    #Get number of clusters
    n_clusters=len(set(labels))
    # Calculate cluster centers
    cluster_centers = Get_centroids(X=X,label=labels)
    # Calculate BCV and WCV
    overall_mean = np.mean(distance_matrix,axis=0)
    BCV = np.sum([np.sum(np.square(cluster_centers[i] - overall_mean)) for i in range(n_clusters)])
    WCV = np.sum([np.sum(np.square(distance_matrix[labels == i] - cluster_centers[i])) for i in range(n_clusters)])
    print(BCV)
    # Calculate CHI
    N = distance_matrix.shape[0]
    CHI = (BCV / WCV) * ((N - n_clusters) / (n_clusters - 1))
    return CHI



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

def Silhouette_analysis(X,cluster_labels,n_clusters,out_dir='./',tag=''):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
   
    silhouette_avg = silhouette_score(X, cluster_labels)
     # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower=10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.\n (Silhouette score {}) ".format(np.round(silhouette_avg,3)))
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = Get_centroids(X=X,label=cluster_labels)
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis  for clustering on sample data with n_clusters = {}".format(n_clusters,),
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(out_dir+tag+'_silhouette_analysis_{}.png'.format(n_clusters))

    


    
    

    
