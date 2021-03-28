# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 08:41:45 2021

@author: james
"""
import os 
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",40)

#%%
if __name__=='__main__':
    df_orig=pd.read_csv('data/Telco-Customer-Churn.csv')
    
    #%%
    # quality of life changes 
    df_orig.columns = df_orig.columns.str.lower() 
    
    
    
    #%%
    #checking missing data
    
    # shows that we can't pick up missing data with isna   
    print(df_orig.isna().any())
    # maybe there are irregular values in the data
    print(df_orig.describe(include='all'))
    # there is no missing data