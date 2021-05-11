# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:03:47 2021

@author: james
"""
"""
Question:
- do I have domain knowledge
-- no! then lets read 
---don't have feedback 
"""
#%%

from sklearn.feature_selection import VarianceThreshold


def binary_feature_selection(dataframe,
                             binary_features,
                             threshold):
    """
    context:
        binary 
    """
    holder = VarianceThreshold(threshold=threshold * (1-threshold))
    dataframe[binary_features] = holder.fit_transform(dataframe[binary_features])
def numerical_feature_selection(dataframe,
                                numerical_features,
                                threshold):
    """
    """
    holder = VarianceThreshold(threshold=threshold * (1-threshold))
    dataframe[numerical_features] = holder.fit_transform(dataframe)
    
# def feature_selection(dataframe):

#%%
# binary_feature_selection(df,
#                          binary_features = b_f,
#                          threshold=)