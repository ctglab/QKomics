import numpy as np
import glob
import pandas as pd
import argparse
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from statistics import mean,variance
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from Kernels.src.kernels_classic import Compute_rbf_kernel
from Kernels.src.Preprocessing import Load_kernels
from Kernels.src.Analysis.Clustering import *
from Kernels.src.Analysis.Kernel import *

#QuAsk
from quask.metrics import calculate_geometric_difference, calculate_generalization_accuracy,calculate_model_complexity ,calculate_kernel_target_alignment


ap=argparse.ArgumentParser()
ap.add_argument('-params','--parameters_file',
                default='hyper_param.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
params_dir=args['parameters_file']


#############################################################################################################################
#                              LOAD DATA AND DEF PARAMETERS                                                                 #
#############################################################################################################################

###########Load hyperparameters from json################
print('Loading Parameters')
# Opening JSON file
f = open(params_dir) 
# returns JSON object as
# a dictionary
params= json.load(f)

res_dir=params['Data']["Output_dir"]

#generate Result dir 
try:
    os.makedirs(res_dir)
except OSError:
    print ("Creation of the directory %s failed. Directory already exist" % res_dir)
else:
    print ("Successfully created the directory %s " % res_dir)


#TO JSON
data_input = pd.read_csv(params['Data']["Input_file"], sep = ",")
n_qubits=params['Kernel']["n_qubits"]
samp_size=params['Data']["Sampling_size"]

#SELECT FT
features=[]
if params['Data']["encoding"]=='separated':
    for i in range(1,int(n_qubits/2) +1):
        name_cna='Component_'+str(i)+'_cna'
        name_exp='Component_'+str(i)+'_exp'
        features.append(name_cna)
        features.append(name_exp)
else:
    for i in range(1,int(n_qubits) +1):
        name_='Component_'+str(i)
        features.append(name_)

labels = 'IntClustMemb'
print(features)

########SAMPLING###################################
df_tot_sel=data_input.sample(n=samp_size,random_state=params['Data']["seed"])
y_train=df_tot_sel[labels].to_numpy()
X_train=df_tot_sel[features]
print(df_tot_sel)
print(y_train)
print(X_train)
    
    

n_samples=len(y_train)
print(n_samples)
#############################################################################################################################
#                              GET PERFORMANCES                                                                             #
#############################################################################################################################
bwidth=params["Scaling"]["bandwidth"]
K=params["Clustering"]["K"]
# create an Empty DataFrame
# object With column names only
df_perf= pd.DataFrame(columns = ['ftmap', 'K', 'Bandwidth','s','geom_distance','concentration','silhouette','Score_cluster','v_intra','v_inter','N_samples'])
df_sil= pd.DataFrame(columns = ['ftmap', 'K', 'Bandwidth','silhouette','N_samples']) 
print(df_perf)

####################################################################################
#                              CLASSICAL KERNEL                                    #
####################################################################################
#get classical kernel
K_classic_tr = Compute_rbf_kernel(X_train,X_train)

#Geometric difference
g_diff=calculate_geometric_difference(K_classic_tr,K_classic_tr)
conc_ck=Kernel_concentration(K_classic_tr)
sc=calculate_model_complexity(K_classic_tr,y_train)

for b in bwidth:
    for k in K:
        #Define clustering
        spectral = SpectralClustering(k, affinity="precomputed",n_init=50,random_state=42)
        cluster_labels = spectral.fit_predict(K_classic_tr)
        score_rbf_4,v_intra_rbf4, v_inter_rbf4=Cluster_score(1-K_classic_tr,cluster_labels)


        #CLUSTER EVALUTION:
        #silhouette score
        sil_rbf_4=silhouette_score(1-K_classic_tr,metric='precomputed',labels=cluster_labels,random_state=42)

        df_perf.loc[len(df_perf)]={'ftmap' : 'rbf', 
                            'K' : k, 
                            'Bandwidth' : b,
                            's':sc,
                            'concentration':conc_ck,
                            'silhouette': sil_rbf_4,
                            'Score_cluster':score_rbf_4 ,
                            'v_intra':v_intra_rbf4,
                            'v_inter':v_inter_rbf4,
                            'geom_distance':g_diff,
                            'N_samples':n_samples}

#############################################################################################################################
#                                                   Clustering                                                              #
#############################################################################################################################    
    
#LOAD KERNELS 
df_new_clust=pd.DataFrame()

##TO JSON
dir=params["Kernel"]["K_dir"]

#CLUSTERING
for i in glob.glob(dir):
    print(i)
    ft_map=i.split('/')[-1]
    for k_dir in sorted(glob.glob(i+'/*')):
        print(k_dir)
    
        q_k_tr=Load_kernels(k_dir=k_dir)
        
        #Compute concentration
        qk_conc=Kernel_concentration(q_k_tr)
        b=k_dir.split('_')[-1].replace('.pickle','')
    

        for k in K:
            #CLUSTERING#
            q_spectral = SpectralClustering(k, affinity="precomputed",random_state=42)
            cluster_labels = q_spectral.fit_predict(q_k_tr)
            
            #Evaluation#
            #From affinaty to distance
            q_k_dist=1-Scale(q_k_tr,(0,1))
            #set diagonal elements to 0
            np.fill_diagonal(q_k_dist,0)
            sil_q=silhouette_score(q_k_dist,metric='precomputed',labels=cluster_labels,random_state=42)
            print(sil_q)
            
            Silhouette_plot(q_k_tr,K,scale=True,out_dir=res_dir,tag='Cluster_'+ft_map+'_'+b+'')
            
            #Score on original data
            #cluster_score = normalized_mutual_info_score(cluster_labels, y_train)
            #Cluster score
            score_q,v_intra_q,v_inter_q=Cluster_score(q_k_dist,cluster_labels)
            

            # #plot new cluster
            # pc_df_4['Cluster_'+ft_map+'_'+b]=cluster_labels
            # g_diff_4=sns.lmplot( x="Component_1_cna", y="Component_1_exp",
            # data=pc_df_4, 
            # palette=sns.color_palette("Paired"),
            # fit_reg=False, 
            # hue='Cluster_'+ft_map+'_'+b, # color by cluster
            # legend=True,
            # scatter_kws={"s": 20}).set(title='METABRIC Specral Clustering for '+ ft_map+'_'+b+ )
            # g_diff_4.savefig(res_dir+'New_clust/Cluster_4'+ft_map+'_'+b+"_.png")
            # plt.close(g_diff_4.fig)
            
            #metrics quask
            qkg_diff=calculate_geometric_difference(K_classic_tr,q_k_tr)
            sq=calculate_model_complexity(q_k_tr,y_train)
            #Add to df
            df_perf.loc[len(df_perf)]={'ftmap' : ft_map, 
                                    'K' : k, 
                                    'Bandwidth' : float(b),
                                    #'cluster_score':cluster_score,
                                    's':sq,
                                    'concentration':qk_conc,
                                    'silhouette':sil_q,
                                    'Score_cluster':score_q ,
                                    'v_intra':v_intra_q,
                                    'v_inter':v_inter_q,
                                    'geom_distance':qkg_diff,
                                    'N_samples':n_samples}
        
    

print(df_perf)
print('##############CHECK##############')
print(df_perf[(df_perf.ftmap.isin(['ZZ_linear','Z_linear'])) & (df_perf.Bandwidth==6.28) & (df_perf.K==4)].concentration)
##SAVE DF##
df_perf.to_csv(res_dir+'clustering_{}_opt_k_reviewed.csv'.format(n_samples))
tt
#############################################################################################################################
#                                                   PLOT                                                                    #
#############################################################################################################################


# Make the PairGrid
g = sns.PairGrid(df_perf.sort_values("K"),
                 x_vars=df_perf.columns[5:-1], y_vars=["Class"],
                 height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h", jitter=False,
      palette="flare_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
#g.set(xlim=(0, 25), xlabel="Crashes", ylabel="")

# Use semantically meaningful titles for the columns
titles = df_perf.columns[5:-1]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)
g.savefig(res_dir+"prova.png")







# #prova=pd.melt(df_perf,id_vars=["Bandwidth","ftmap","K"],value_vars=["test_error","train_error"])
# #print(prova)
# prova.rename(columns={'variable':'train/test','value':'accuracy'},inplace=True)
# Draw a pointplot to show pulse as a function of three categorical factors
g4 = sns.catplot(data=df_perf, x="Bandwidth", y="cluster_score", hue="ftmap", col="K",
                capsize=.2, palette="YlGnBu_d",kind="point")

g4.despine(left=True)
g4.set(xlabel='Max angle(rad)')
#sns.move_legend(g, "lower right", bbox_to_anchor=(0, 1))
g4.savefig(res_dir+"Scores/cluster_score.png") 

g_sil = sns.catplot(data=df_perf, x="Bandwidth", y="silhouette", hue="ftmap", col="K",
                capsize=.2, palette="YlGnBu_d",kind="point")

g_sil.despine(left=True)
g_sil.set(xlabel='Max angle(rad)')
#sns.move_legend(g, "lower right", bbox_to_anchor=(0, 1))
g_sil.savefig(res_dir+"Scores/cluster_silhouette.png") 

g_score = sns.catplot(data=df_perf, x="Bandwidth", y="Score_cluster", hue="ftmap", col="K",
                capsize=.2, palette="YlGnBu_d",kind="point")

g_score.despine(left=True)
g_score.set(xlabel='Max angle(rad)')
#sns.move_legend(g, "lower right", bbox_to_anchor=(0, 1))
g_score.savefig(res_dir+"Scores/clustering_score_intra_inter.png") 

#MELT FOR VARIANCE
df_var=pd.melt(df_perf,id_vars=["Bandwidth","ftmap","K"],value_vars=["v_inter","v_intra"])
print(df_var)

df_var.rename(columns={'variable':'intra/inter','value':'var'},inplace=True)
g_var = sns.catplot(data=df_var, x="Bandwidth", y='var', hue="ftmap", col="K",row='intra/inter',
                capsize=.2, palette="YlGnBu_d",kind="point")

g_var.despine(left=True)
g_var.set(xlabel='Max angle(rad)')
#sns.move_legend(g, "lower right", bbox_to_anchor=(0, 1))
g_var.savefig(res_dir+"Scores/clustering_var_intra_inter.png") 