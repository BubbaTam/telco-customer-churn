# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:10:40 2021

@author: james
"""
#%%
import os 
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",40)
#%%
#os.getcwd()
path = r'Telco customer churn'
os.chdir(path)
#%%
if __name__ == '__main__':
    dataframe_orig = pd.read_csv('data/Telco-Customer-Churn.csv')
    data
    
    
    print(dataframe_orig.head(5))
    print(dataframe_orig.shape)
    
    #%%
    # will want to have an initial 
    print(dataframe_orig.columns)
    
    
    
    
    #%% 
    print(dataframe_orig.isna().sum())
    #no missing data
    info = dataframe_orig.info()
    print(info)
