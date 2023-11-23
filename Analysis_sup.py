import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import argparse
import pandas as pd
import json 

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from Kernels.src.kernels_classic import *
from Kernels.src.Preprocessing import Load_kernels,Split_and_sample
from Kernels.src.Analysis.Kernel import *



#QuAsk
from quask.metrics import calculate_geometric_difference, calculate_generalization_accuracy,calculate_model_complexity ,calculate_kernel_target_alignment


#############################################################################################################################
#                                         ARGPARSE +JSON                                                                       #
#############################################################################################################################
ap=argparse.ArgumentParser()
ap.add_argument('-params','--parameters_file',
                default='hyper_param.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
params_dir=args['parameters_file']

# Opening JSON file
f = open(params_dir) 
# returns JSON object as
# a dictionary
params= json.load(f)
#############################################################################################################################
#                              LOAD DATA AND SET PARAMETERS                                                                 #
#############################################################################################################################
#TO JSON
input_file=params['Data']["Input_file"]
sampling_sz=params['Data']["Sampling_size"]
output_dir=params['Data']["Output_dir"]
task=list(params['Data']["task"])[0]
n_qubits=params['Kernel']["n_qubits"]
bandwidth=params['Scaling']['bandwidth']
maps=params['ft_maps']['maps']
ker_dir=params['Kernel']["K_dir"]

#generate Result dir 
try:
    os.makedirs(output_dir)
except OSError:
    print ("Creation of the directory %s failed. Directory already exist" % output_dir)
else:
    print ("Successfully created the directory %s " % output_dir)


# load data and sample
data_input = pd.read_csv(input_file, sep = ",")
data_input=data_input.sample(n=sampling_sz,axis=0,random_state=42)


#SELECT FT
features=[]
for i in range(1,int(n_qubits/2) +1):
    name_cna='Component_'+str(i)+'_cna'
    name_exp='Component_'+str(i)+'_exp'
    features.append(name_cna)
    features.append(name_exp)
labels = 'IntClustMemb'



#Preprocess according to task
data_dict={}

if task=='Supervised':
    tr_sz=params['Data']["task"][task]['tr_sz']
    ts_sz=params['Data']["task"][task]['ts_sz']
    balanced=params['Data']["task"][task]['balanced']
    min_sz=params['Data']["task"][task]['min_sz']
    for case,class_ in params['Data']["task"][task]['classes'].items():
        print(case)
        df_tot_sel=df_tot_sel=data_input.loc[data_input.IntClustMemb.isin(class_)]
        #
        X_train,y_train,X_test,y_test=Split_and_sample(df_tot_sel,
                                                       features,labels,
                                                       tr_sz=tr_sz,ts_sz=ts_sz,
                                                        min_sz=min_sz)
        data_dict[case]={}
        data_dict[case]['X_train']=X_train
        data_dict[case]['X_test']=X_test
        data_dict[case]['y_train']=y_train
        data_dict[case]['y_test']=y_test
        print('{} train = {} samples '.format(case,y_train.shape))
        print('{} test = {} samples '.format(case,y_test.shape))

#############################################################################################################################
#                              GET PERFORMANCES                                                                             #
#############################################################################################################################

# create an Empty DataFrame
# object With column names only
df_perf= pd.DataFrame(columns = ['ftmap', 'N_classes', 'Bandwidth','test_error','train_error','s','geom_distance','concentration','optimize','C'])
print(df_perf)

####################################################################################
#                              CLASSICAL KERNEL                                    #
####################################################################################

for case_ in data_dict.keys():
        print('----------CASE {}-------'.format(case_),flush=True)
        X_train=data_dict[case_]['X_train']
        X_test=data_dict[case_]['X_test']
        y_train=data_dict[case_]['y_train']
        y_test=data_dict[case_]['y_test']
        #Compute Kernel
        K_classic_tr= Compute_rbf_kernel(X_train,X_train)
 
        #Geometric difference
        g_diff=calculate_geometric_difference(K_classic_tr,K_classic_tr)
        conc_ck=Kernel_concentration(K_classic_tr)
        sc=calculate_model_complexity(K_classic_tr,y_train)
        #Use SVM
        normal_svc= SVC(kernel = "rbf")
        normal_svc.fit(X_train, y_train)
        #Test SVC 10
        y_train_pred=normal_svc.predict(X_train)
        y_test_pred=normal_svc.predict(X_test)
        for b in bandwidth:
            df_perf.loc[len(df_perf)]={'ftmap' : 'rbf', 
                                    'N_classes' : case_, 
                                    'Bandwidth' : np.round(float(b)*np.pi,2),
                                    'test_error':accuracy_score(y_test, y_test_pred),
                                    'train_error':accuracy_score(y_train, y_train_pred),
                                    's':sc,
                                    'concentration':conc_ck,
                                    'geom_distance':g_diff,
                                    'optimize':False,
                                    'C':1.0}
            

       
QSVC =SVC(kernel='precomputed')

#
#Loop over cases,ft maps, and scaling
for key in maps.keys():
    #get ft maps params
    ft=maps[key]['ft_map']
    ent=maps[key]['ent_type']
    ft_map=ft+'_'+ent

    print('#############{}_{}################'.format(ft,ent))
    
    for case_ in data_dict.keys():
        print('----------CASE {}-------'.format(case_))
        X_train=data_dict[case_]['X_train']
        X_test=data_dict[case_]['X_test']
        y_train=data_dict[case_]['y_train']
        y_test=data_dict[case_]['y_test']
        #Compute Kernel
        K_classic_tr= Compute_rbf_kernel(X_train,X_train)
        for b in bandwidth:
            print('Bandwidth {}'.format(b))
            
            k_dir_tr=ker_dir+'/'+case_+'/'+ft+'_'+ent+'/'+'qk_tot_tr_{}.pickle'.format(b)
            
            #Load tr
            q_k_tr=Load_kernels(k_dir=k_dir_tr)
            print('X_train shape : {}'.format(X_train.shape))
            print('K_train shape : {}'.format(q_k_tr.shape))
            print('Concentration')
            qk_conc=Kernel_concentration(q_k_tr)
            print('Load ts')
            k_dir_ts=k_dir_tr.replace('tr','ts')
            q_k_ts=Load_kernels(k_dir=k_dir_ts)
            
            print('## QSVM ##')
            QSVC.fit(q_k_tr, y_train)
            pred_labels_tr =QSVC.predict(q_k_tr)
            pred_labels_ts =QSVC.predict(q_k_ts)
            print('#metrics quask')
            qkg_diff=calculate_geometric_difference(K_classic_tr,q_k_tr)
            sq=calculate_model_complexity(q_k_tr,y_train)
            print('#Add to df')
            df_perf.loc[len(df_perf)]={'ftmap' : ft_map, 
                                    'N_classes' : case_, 
                                    'Bandwidth' : np.round(float(b)*np.pi,2),
                                    'test_error':accuracy_score(y_test, pred_labels_ts),
                                    'train_error':accuracy_score(y_train, pred_labels_tr),
                                    's':sq,
                                    'concentration':qk_conc,
                                    'geom_distance':qkg_diff,
                                    'optimize':False,
                                    'C':1.0}
print('#################  OPTIMIZE ############################')
parameters={'C':[0.1,1,10,100]}
for key in maps.keys():
    #get ft maps params
    ft=maps[key]['ft_map']
    ent=maps[key]['ent_type']
    ft_map=ft+'_'+ent

    print('#############{}_{}################'.format(ft,ent))
    
    for case_ in data_dict.keys():
        print('----------CASE {}-------'.format(case_))
        X_train=data_dict[case_]['X_train']
        X_test=data_dict[case_]['X_test']
        y_train=data_dict[case_]['y_train']
        y_test=data_dict[case_]['y_test']
        #Compute Kernel
        K_classic_tr= Compute_rbf_kernel(X_train,X_train)
        for b in bandwidth:
            print('Bandwidth {}'.format(b))
            
            k_dir_tr=ker_dir+'/'+case_+'/'+ft+'_'+ent+'/'+'qk_tot_tr_{}.pickle'.format(b)
            
            #Load tr
            q_k_tr=Load_kernels(k_dir=k_dir_tr)
            print('X_train shape : {}'.format(X_train.shape))
            print('K_train shape : {}'.format(q_k_tr.shape))
            print('Concentration')
            qk_conc=Kernel_concentration(q_k_tr)
            print('Load ts')
            k_dir_ts=k_dir_tr.replace('tr','ts')
            q_k_ts=Load_kernels(k_dir=k_dir_ts)
            
            print('## QSVM ##')

            clf_opt = GridSearchCV(QSVC, parameters)
            clf_opt.fit(q_k_tr,y_train )
            pred_labels_tr =clf_opt.predict(q_k_tr)
            pred_labels_ts =clf_opt.predict(q_k_ts)
            print('#metrics quask')
            qkg_diff=calculate_geometric_difference(K_classic_tr,q_k_tr)
            sq=calculate_model_complexity(q_k_tr,y_train)
            print('#Add to df')
            df_perf.loc[len(df_perf)]={'ftmap' : ft_map, 
                                    'N_classes' : case_, 
                                    'Bandwidth' : np.round(float(b)*np.pi,2),
                                    'test_error':accuracy_score(y_test, pred_labels_ts),
                                    'train_error':accuracy_score(y_train, pred_labels_tr),
                                    's':sq,
                                    'concentration':qk_conc,
                                    'geom_distance':qkg_diff,
                                    'optimize':True,
                                    'C':clf_opt.best_params_['C']}


for case_ in data_dict.keys():
        print('----------CASE {}-------'.format(case_),flush=True)
        X_train=data_dict[case_]['X_train']
        X_test=data_dict[case_]['X_test']
        y_train=data_dict[case_]['y_train']
        y_test=data_dict[case_]['y_test']
        #Compute Kernel
        K_classic_tr= Compute_rbf_kernel(X_train,X_train)
 
        #Geometric difference
        g_diff=calculate_geometric_difference(K_classic_tr,K_classic_tr)
        conc_ck=Kernel_concentration(K_classic_tr)
        sc=calculate_model_complexity(K_classic_tr,y_train)
        #Use SVM
        normal_svc= SVC(kernel = "rbf")
        clf_opt_c = GridSearchCV(normal_svc, parameters)
        clf_opt_c.fit(X_train,y_train )
        #Test SVC 10
        y_train_pred=clf_opt_c.predict(X_train)
        y_test_pred=clf_opt_c.predict(X_test)
        for b in bandwidth:
            df_perf.loc[len(df_perf)]={'ftmap' : 'rbf', 
                                    'N_classes' : case_, 
                                    'Bandwidth' : np.round(float(b)*np.pi,2),
                                    'test_error':accuracy_score(y_test, y_test_pred),
                                    'train_error':accuracy_score(y_train, y_train_pred),
                                    's':sc,
                                    'concentration':conc_ck,
                                    'geom_distance':g_diff,
                                    'optimize':True,
                                    'C':clf_opt_c.best_params_['C']}
            





print(df_perf)
#TO JSON
df_perf.to_csv(output_dir+'Supervised_results_{}.csv'.format(len(data_input)))
tt
#############################################################################################################################
#                                                   KERNEL ALIGNMENT                                                        #
#############################################################################################################################    
    


#############################################################################################################################
#                                                   PLOT                                                                    #
#############################################################################################################################
sns.set_theme(style="whitegrid")

df_perf.replace('Z_linear','Z')
prova=pd.melt(df_perf,id_vars=["Bandwidth","ftmap","N_classes"],value_vars=["test_error","train_error"])
print(prova)
prova.rename(columns={'variable':'train/test','value':'accuracy'},inplace=True)
# Draw a pointplot to show pulse as a function of three categorical factors

g4 = sns.catplot(data=prova.loc[prova.N_classes=='4_classes'].sort_values(by='train/test',ascending=False), x="Bandwidth", y="accuracy", hue="ftmap", row="N_classes",
                col='train/test',capsize=.2, palette="YlGnBu_d",kind="point")

g4.despine(left=True)
g4.set(xlabel='Max angle(rad)')
#sns.move_legend(g, "lower right", bbox_to_anchor=(0, 1))
g4.savefig("sup_confront_4.png") 

#Kernel concentration 
g = sns.catplot(
    data=df_perf, x="Bandwidth", y="concentration", hue="ftmap", col="N_classes",
    capsize=.2, palette="YlGnBu_d",
    kind="point", height=6, aspect=.75,
)
g.set(ylabel='Kernel variance',xlabel='Max angle(rad)')
g.despine(left=True)
g.fig.suptitle('Kernel non-diagonal entries distribution',x=0.5,y=1)

g.savefig("K_concentration_confront.png") 