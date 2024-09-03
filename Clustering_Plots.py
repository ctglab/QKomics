import numpy as np
import pickle
import json
import pandas as pd
import argparse 
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score,mutual_info_score,normalized_mutual_info_score
import matplotlib.pyplot as plt
#get classical kernel
from sklearn.preprocessing import MinMaxScaler
import os 
from scipy.stats import entropy
from Kernels.src.Preprocessing import Load_kernels
from Kernels.src.kernels_classic import Compute_rbf_kernel

def Parse_data(df):
    #Replace
    df.replace('Z_full','Z',inplace=True)
    df.rename(columns={'Bandwidth':'Max angle'},inplace=True)
    df['feature maps']=df['ftmap'].astype('category')
    df['Max angle']=df['Max angle']*np.pi
    df['Max angle']=df['Max angle'].round(2)
    return df

def Plot_trend(df,score,out_dir):
    g=sns.lineplot(data=df,
             x="Max angle",
             y=score,
             hue='feature maps')
    plt.title('{} trend {} samples '.format(score,df.N_samples[0]))
    plt.xlabel('max angle range(rad)')
    #plt.ylim(0,0.4)
    plt.xticks(df['Max angle'].unique(),rotation=50)
    sns.move_legend(g, "upper right")
    plt.savefig(out_dir+'/clustering_{}_trend.png'.format(score))
    plt.close()
    return 0

def Plot_score(df,score,out_dir):
    sns.set_theme(style="whitegrid")

    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    g = sns.relplot(
        data=df,
        style="Max angle", x=score,
        hue="feature maps",
        y="K",s=100,
        sizes=(10, 200),
        height=6, aspect=1.25,
              
    )

    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    #g.ax.set_xlim(-0.0,0.4)
    #g.despine(left=True, bottom=True)


    g.set(title='{} across k ({} samples)'.format(score,df.N_samples[0]))
    #plt.title()
    #plt.xlabel('Number of Clusters')
    # Show the plot
    #plt.tight_layout()
    sns.move_legend(g, "upper right",bbox_to_anchor=(1.0,0.8), ncol=1)
    plt.savefig(out_dir+'/opt_k_{}.png'.format(score))
    plt.close()
    return 0

def Plot_score_heatmap(df,score,out_dir):
   
    
    #map bandwidth
    df[r'$\beta$']=df['Max angle'].map({0.125:r'$\frac{\pi}{8}$',0.25:r'$\frac{\pi}{4}$',0.5:r'$\frac{\pi}{2}$',1:r'$\pi$',2:r'$2\pi$'})
    #map rbf beta to 1
    df.loc[df.ftmap=='rbf',r'$\beta$']='1'
    label_x=[r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\pi$', r'$2\pi$']
    cmap_new=sns.color_palette("Reds", as_cmap=True)
    fig, axs = plt.subplots(1, 4, figsize=(18, 5),gridspec_kw={'width_ratios': [1,3,3,3]})
    sns.set_theme(style="whitegrid")

    for i, ftmap in enumerate(['rbf','Z','ZZ_linear','ZZ_full']):
        print(ftmap)

        # Pivot the DataFrame
        new_df = df[df.ftmap==ftmap].pivot(index='K', columns=r'$\beta$', values='silhouette')
        if ftmap!='rbf':
            new_df=new_df.reindex(label_x,axis=1)
        
        # Disable LaTeX interpreter
        plt.rcParams['text.usetex'] = False
        
        # Plot the heatmap in the corresponding subplot
        ax = axs[i]
        sns.heatmap(new_df, annot=True, linewidth=.5, vmin=0, vmax=0.65, cmap=cmap_new, ax=ax)
        
        # Set the subplot title
        ax.set_title(ftmap)
        
        # Set the x and y labels
        ax.set(xlabel=r'$\beta$', ylabel='K')
        
        # Set the x tick labels
    # ax.set_xticklabels(label_x, rotation=50, fontsize=10)

    # Adjust the layout    
    plt.tight_layout()
    #save
    plt.savefig(out_dir+'heatmap_{}.png'.format(score),dpi=300)
    plt.close()
    return 0



#Parser
ap=argparse.ArgumentParser()
ap.add_argument('-params','--parameters_file',
                default='hyper_param.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
params_dir=args['parameters_file']

print('Loading Parameters')
# Opening JSON file
f = open(params_dir) 
# returns JSON object as
# a dictionary
params= json.load(f)

results_to_process=params['to_process']
scores=params['Scores']

for path in results_to_process:
    df=pd.read_csv(path)
    outdir=path.split('/')[:-1]
    outdir='/'.join(outdir)  
    print(outdir)
    #Preproces
    df=Parse_data(df=df) 
    print(df.head())
   
    for sc in scores:
        print(sc)
        if sc in df.columns:
            #Plot trend
            Plot_trend(df=df,score=sc,out_dir=outdir)
            #Plot score opt
            Plot_score(df=df,score=sc,out_dir=outdir)
            #Plot heatmap
            Plot_score_heatmap(df=df,score=sc,out_dir=outdir)
        else:
            print('{} not in df'.format(sc))


