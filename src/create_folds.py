# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:10:40 2021

@author: james
"""
#%%
#standard library imports 
import os 
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",40)
from sklearn import model_selection
#related thrid party imports 

#local application/livrary specific imports
#%%
#os.getcwd()
#path = r'Telco customer churn'
#os.chdir(path)
#%%
if __name__ == '__main__':
    # importing the csv file 
    df = pd.read_csv('data/Telco-Customer-Churn-clean.csv')
    
    #create a new column
    df["k_fold"] = -1

    # identify the target variable 
    y = df.churn.values 
    
    # making the folds 
    skf = model_selection.StratifiedKFold(n_splits=5,shuffle=True,
                                          random_state=42)
    
    # fill in the k_fold column
    # for each 'fold' we get a list of indicies for the training 't_' and 
    # validation
    for fold, (t_,v_) in enumerate(skf.split(X=df,y=y)):
        df.loc[v_,'k_fold']= fold  #works because loc[rows,column]
    df.to_csv('data/telco-customer-churn-clean-folds.csv',index=False)