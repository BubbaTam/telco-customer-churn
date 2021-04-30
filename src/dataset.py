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
from sklearn.compose import ColumnTransformer
#local application/library specific imports
#import src.utils as utils
#%%
os.chdir(r'C:\Users\james\OneDrive\Desktop\Data Science\modelling\Churn prediction\Telco customer churn')
print(os.getcwd())


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
binary_features = ['gender','senior_citizen','partner','dependents','phone_service',
                   'paperless_billing']
nominal_features = ['multiple_lines', 'internet_service','online_security',
                    'online_backup','device_protection','tech_support',
                    'streaming_tv','streaming_movies','contract',
                    'payment_method'
                    ]

numerical_features = ['tenure','monthly_charges','total_charges']
target_feature =['churn']
fold_feature = ['k_fold']
#sorting_categorical_features(df,binary_features,nominal_features)

#%%
def sorted_categorical_features(dataframe,
                                binary_features = None,
                                ordinal_features = None,
                                ordinal_features_order = 'auto',
                                nominal_features = None,
                                target_feature=None):
    """
    prior:
        - the data is cleared
    """
    
    df_2 = dataframe.copy()
    # Encode the target feature
    df_2.loc[:,target_feature] = LabelEncoder().fit_transform(df_2[target_feature].values)
    #df_2.loc[:,target_feature] = df_2.loc[:,target_feature].values.ravel()
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
    
    
   



#%%
df_sorted_cf = sorted_categorical_features(df,
                                           binary_features = binary_features,
                                           ordinal_features = None,
                                           ordinal_features_order = 'auto',
                                           nominal_features = nominal_features,
                                           target_feature = target_feature)
  #%%
print(sorted(df_sorted_cf.columns))
print('***')
print(sorted(df.columns))
#%%
pete = df[binary_features+
          nominal_features+
          numerical_features+
          target_feature+
          fold_feature]

for f in df.columns:
    if f not in pete.columns:
        print(f )