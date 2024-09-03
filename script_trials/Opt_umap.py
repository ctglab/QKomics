import numpy as np
#from sklearn.preprocessing import MinMaxScaler

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from Kernels.src.Analysis.utils import Score_cluster

from sklearn.decomposition import TruncatedSVD, PCA
import matplotlib.pyplot as plt
import seaborn as sns
import umap 
 
#Set seed
seed=42
###############
#Expression
data_exp=pd.read_csv("/CTGlab/data/brca_metabric/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",sep='\t')
#CNV
data_cnv=pd.read_csv("/CTGlab/data/brca_metabric/data_cna.txt",sep='\t')
#Clin
data_clin=pd.read_csv("/CTGlab/data/brca_metabric/data_clinical_sample.txt",sep='\t',skiprows=4)
data_clin.set_index('SAMPLE_ID',inplace=True)

#bind index colums
data_exp['hugo_entrez_combo']=data_exp.Hugo_Symbol+'_'+data_exp.Entrez_Gene_Id.astype(str)
data_cnv['hugo_entrez_combo']=data_cnv.Hugo_Symbol+'_'+data_cnv.Entrez_Gene_Id.astype(str)
data_exp.drop(['Hugo_Symbol','Entrez_Gene_Id'],axis=1,inplace=True)
data_cnv.drop(['Hugo_Symbol','Entrez_Gene_Id'],axis=1,inplace=True)
#Samples present in both dataset
samples=list(set(data_exp.columns) & set(data_cnv.columns))
data_exp_sub=data_exp[samples]
data_cnv_sub=data_cnv[samples]
samples.remove('hugo_entrez_combo')
data_clin=data_clin[data_clin.index.isin(samples)]

gene_com_id=list(set(data_exp_sub.hugo_entrez_combo) & set(data_cnv_sub.hugo_entrez_combo))
print('common features: ',len(gene_com_id))

#SET hugo+entrez as index
data_exp_sub.set_index('hugo_entrez_combo',inplace=True)
data_cnv_sub.set_index('hugo_entrez_combo',inplace=True)
#drop nan
if data_exp_sub.isnull().values.any():
    data_exp_sub=data_exp_sub.apply(lambda x: x.fillna(x.mean()),axis=0)

if data_cnv_sub.isnull().values.any():
    data_cnv_sub=data_cnv_sub.apply(lambda x: x.fillna(x.mean()),axis=0)

data_clin_patient=pd.read_csv('/CTGlab/data/brca_metabric/data_clinical_patient.txt',sep='\t',skiprows=4,index_col='PATIENT_ID')
ic10=data_clin_patient['INTCLUST']
ic10.dropna(inplace=True)
ic10.unique()
ic10.replace('4ER+','4',inplace=True)
ic10.replace('4ER-','4',inplace=True)
## MERGE exp & cnv
data_full=data_exp_sub.T.join(data_cnv_sub.T,how='inner')

####### OPT UMAP ###########
ic10=ic10.reindex(data_full.index)

n_neighbour=[5,10,15,30,50]
min_d=[0.05,0.1,0.25]
spread=[0.25,0.5,1]
df_opt=pd.DataFrame(columns=['n_neighbour','min_d','spread','clustering_score'])
tot=len(min_d)*len(spread)*len(n_neighbour)
i=1
for n in n_neighbour:
    for d in min_d:
        for s in spread:
            print("####### {} out of {} ######".format(i,tot))
            mapper=umap.UMAP(n_components=4,n_neighbors=n,min_dist=d,spread=s,random_state=seed).fit(data_full)
            data_umap= mapper.transform(data_full)
            # Score
            score=silhouette_score(data_umap,ic10,random_state=seed)
            print('Score : ',score)
            #Save values
            df_opt.loc[len(df_opt)]={
                'n_neighbour':n,
                'min_d':d,
                'spread':s,
                'clustering_score':score
            }
            i+=1
df_opt.to_csv('opt_umap.csv')