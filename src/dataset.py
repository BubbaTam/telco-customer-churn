# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 08:41:45 2021

@author: james
"""
#%%
"""
That the data has come in a static format, where all the training and testing 
data is in one csv file. 
"""
#%%
#standard library imports 
import os 

#related third party imports 
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",60)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
#local application/library specific imports
import src.utils as utils
import src.dispatcher as dispatcher

#%%
def sorting_categorical_features(df,binary_features,
                                 nominal_features):
    """
    return a new dataframe with categorical variables encoded
    """
    new_df = df.copy()
    # Encode the target feature
    for column in binary_features:
        new_df.loc[:,column] = LabelEncoder().fit_transform(new_df[column].values)
    ohe = OneHotEncoder(categories='auto',drop=None,sparse=False,dtype=int)
    ohe.fit_transform(new_df)
    print(ohe.categories)
    print(new_df)
    return new_df
#%%
df = pd.read_csv('data/telco-customer-churn-clean-folds.csv')
#%%


#%%
def sorted_categorical_features_df(dataframe,
                                   binary_features = None,
                                   ordinal_features = None,
                                   ordinal_features_order = 'auto',
                                   numerical_features = None,
                                   nominal_features = None,
                                   target_feature=None,
                                   fold_feature = None
                                   ):
    """
    prior:
        - the data is cleaned
        - no missing data
    points of contention:
        - the use of labelencoder so will just use ordinalencoder
    options:
        - create a new dataframe with all featues 
        - to return the input and target feature/s separately 
    """
    
    df_2 = dataframe.copy()
    # Encode the target feature
    enc = OrdinalEncoder(dtype = int)
    df_2.loc[:,target_feature] = enc.fit_transform(df_2.loc[:,target_feature])
    
    
    ###if i was to split the data from here into input and target variable/s
    #df_2.loc[:,target_feature] = LabelEncoder().fit_transform(df_2[target_feature].values)
    
    
    
    # Encode binary input features 
    if binary_features != None:
        for column in binary_features:
            df_2.loc[:,column] = LabelEncoder().fit_transform(df_2[column].values)
            
            
    # Encode ordinal input features    
    if ordinal_features != None:
        enc = OrdinalEncoder(categories=ordinal_features_order,
                              dtype = int
                              )
        df_2[ordinal_features] = enc.fit_transform(df_2[ordinal_features])
        df_2[ordinal_features] = enc.fit_transform(df_2[ordinal_features])
        
                   
    # Encode nominal input features 
    if nominal_features != None:
        ohe = OneHotEncoder(drop=None,
                            sparse=False,
                            dtype=int)
        nominal = ohe.fit_transform(df_2[nominal_features])
        nominal_df = pd.DataFrame(data = nominal,
                                  columns = ohe.get_feature_names()
                                  )
    
    df_sorted_cf = pd.concat([df_2[
                                   binary_features+
                                   numerical_features
                                   ],
                              nominal_df,
                              df_2[
                                  target_feature+
                                  fold_feature]],
                              axis=1)
    
    return df_sorted_cf

def standardise_numerical_features(df,numerical_features):
    df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])
def normalise_numerical_features(df,numerical_features):
    df[numerical_features] = Normalizer().fit_transform(df[numerical_features])
def identify_input_features(binary_features,
                            ordinal_features,
                            numerical_features,
                            nominal_features,
                            target_feature,
                            fold_feature):
    """
    Default is that target_feature and fold_feature are not included.
    The function identifies datatypes that are not present and excludes them  
    """
    not_input_features = [target_feature,fold_feature]
    [not_input_features.append(i) for i in dispatcher.features if dispatcher.features[i] == None]
    input_features = [i for i in dispatcher.features if i not in not_input_features] 
    return input_features
     
   
