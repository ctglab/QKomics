import numpy as np
import pandas as pd
import argparse
import os
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler







def Sample(data_input,features,labels,samp_size,):
    """
    Split 
    """
    df_tot_sel=data_input.sample(samp_size,random_state=89)
    y_train=df_tot_sel[labels]
    X_train=df_tot_sel[features]
    return X_train,y_train




def Split_and_sample(df_tot_sel,features,labels,
                     tr_sz=0.6,ts_sz=0.4,
                     min_sz=70):
    """
    Take dataframe split in test and train and sample balancing classes,returns X_train/test and y_train/test

    """
    #Extract data and Split train and test
    X_train, X_test, y_train, y_test = train_test_split(df_tot_sel[features],df_tot_sel[labels],
                                                        train_size=tr_sz,test_size=ts_sz,
                                                        random_state=89,stratify=df_tot_sel[labels],balanced=True)
            

    print(X_train.shape)
    ########BALANCED SAMPLING BY CLASS###################################
    if balanced ==True:
        X_train['Class']=y_train
        g_tr = X_train.groupby('Class')
        print("Train class dist:")
        print(g_tr.size())
        samp_size_tr=min(min_sz,g_tr.size().min())
        X_train= pd.DataFrame(g_tr.apply(lambda x: x.sample(samp_size_tr,random_state=89)))
        print(type(X_train))
        y_train=X_train['Class'].droplevel(level='Class')
        X_train=X_train[features].droplevel(level='Class')

        #test
        X_test['Class']=y_test
        g_ts = X_test.groupby('Class')
        print("Test class dist:")
        print(g_ts.size())
        samp_size_ts=min(min_sz,g_ts.size().min())
        #X_test= pd.DataFrame(g_ts.apply(lambda x: x.sample(ts_size,random_state=89)))
        X_test= pd.DataFrame(g_ts.apply(lambda x: x.sample(samp_size_ts,random_state=89)))
        y_test=X_test['Class'].droplevel(level='Class')
        X_test=X_test[features].droplevel(level='Class')

    #Transform in numpy obj
    y_train= y_train.to_numpy()
    y_test= y_test.to_numpy()

    return X_train,y_train,X_test,y_test

