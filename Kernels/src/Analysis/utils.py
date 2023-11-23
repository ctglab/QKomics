import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


#Get centroids of N clusters################################
#X:array(N_samples,N_features)                             #
#label:array/list (N_samples) with cluster                 #
#returns array(N_cluster,N_feature) with centroids coord   #
############################################################
def Get_centroids(X,label):
   label=np.asarray(label)
   classes=np.unique(label)
   centroids_=[]
   for i in range(0,len(classes)):
      c_=sum(X[tuple([label==classes[i]])])/len(label[tuple([label==classes[i]])])
      centroids_.append(c_)
   centroids=np.vstack(centroids_) 
   return centroids

#Get Intra cluster dist of N clusters#######################
#X:array(N_samples,N_features)                             #
#label:array/list (N_samples) with cluster                 #
#returns list(N_cluster) with intra cluster distances      #
############################################################
def Intra_clust_dist(X,label):
    centroids=Get_centroids(X,label=label)
    label=np.asarray(label)
    classes=np.unique(label)
    distances=[]
    for i in range(0,len(classes)):
        intra_=euclidean_distances(centroids[i].reshape(1, -1),X[tuple([label==classes[i]])])
        distances.append(intra_)
    return(distances)
#Get Inter cluster dist of N clusters#######################
#X:array(N_samples,N_features)                             #
#label:array/list (N_samples) with cluster                 #
#returns list(N_cluster) with inter cluster distances      #
############################################################
def Inter_clust_dist(X,label):
    centroids=Get_centroids(X,label=label)
    label=np.asarray(label)
    classes=np.unique(label)
    distances=[]
    for i in range(0,len(classes)):
        inter_=euclidean_distances(centroids[i].reshape(1, -1),X[tuple([label!=classes[i]])])
        distances.append(inter_)
    return(distances)

#Evaluate clusters##########################################
#X:array(N_samples,N_features)                             #
#label:array/list (N_samples) with cluster                 #
#returns max_intra_clust-dist/min_inter_clust_dist,        #
# a measure of how much point are well clustered           #
# the smaller the better(<1)                               #
############################################################
def Score_cluster(X,label):
    centroids=Get_centroids(X,label=label)
    label=np.asarray(label)
    classes=np.unique(label)
    inter_=Inter_clust_dist(X,label)
    intra_=Intra_clust_dist(X,label)
    scores=[]
    for i in range(0,len(classes)):
        max_intra=intra_[i].max()
        min_inter=inter_[i].min()
        s=max_intra/min_inter
        scores.append(s)
    return scores