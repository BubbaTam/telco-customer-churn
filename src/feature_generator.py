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

from sklearn.feature_selection import VarianceThreshold,chi2,mutual_info_classif,mutual_info_regression, SelectKBest, f_regression,f_classif
import src.dispatcher as dispatcher

#%%
def feature_engineering(dataframe):
    dataframe.loc[:,'engaged']=1 
    dataframe.loc[(dataframe['contract']=='Month-to-month'),'engaged']=0

    dataframe.loc[:,'y_and_not_e']=0
    dataframe.loc[(dataframe['senior_citizen']==0) & (dataframe['engaged']==0),'y_and_not_e']=1
        
    #dataframe.loc[:,'elect_check']=0 
    #dataframe.loc[(dataframe['payment_method']=='electronic_check') & (dataframe['engaged']==0),'elect_check']=1
        
    dataframe.loc[:,'fiber_opt']=1 
    dataframe.loc[(dataframe['internet_service']!='Fiber optic'),'fiber_opt']=0
        
    dataframe.loc[:,'stream_no_int']=1 
    dataframe.loc[(dataframe['streaming_tv']!='No internet service'),'stream_no_int']=0
        
    dataframe.loc[:,'no_prot']=1 
    dataframe.loc[(dataframe['online_backup']!='No') | (dataframe['device_protection']!='No') | (dataframe['tech_support']!='No'),'no_prot']=0
        
    dataframe['total_services'] = (dataframe[['phone_service', 'internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']]== 'Yes').sum(axis=1)

def manual_feature_reduction(dataframe,
                             *args):
    """
    """
    # redundant features
    # I already know about the same feature multiple times
    #There was a 1:1 relationship with all 
    internet_service_features = ['x7_No internet service',
                                 'x6_No internet service',
                                 'x5_No internet service',
                                 'x4_No internet service',
                                 'x3_No internet service',
                                 'x2_No internet service'
                                  ]
    for _ in internet_service_features:
        dataframe = dataframe.drop(_,axis=1)
    dataframe = dataframe.drop('x0_No phone service',axis=1)


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
    
    
def feature_selection(dataframe,
                      feature_selection_correlation,
                      type_of_ml_problem = None,
                      ordinal_features = None,
                      nominal_features = None,
                      numerical_features = None,
                      target_features = None):
    """
    when:
        input(numerical) : target (numerical):
            Pearson's correlation (linear)
            Spearman's Rank coefficient (nonlinear)
        input(numerical) : target(categorical):
            ANOVA correlation coefficient (linear)
            Kendall's rank coefficient (nonlinear)
        input (categorical) : target (numerical):
            reverse of above
        Input (categorical) : target (categorical):
            Chi-Squared test
            Mutual information   				
    """
    if feature_selection_correlation == "pearson":
        corr = dataframe[numerical_features].corr("pearson")
        print(corr)
    elif feature_selection_correlation == 'spearman':
         corr = dataframe[numerical_features].corr('spearman')
         print(corr)
    elif feature_selection_correlation == 'ANOVA':
        if type_of_ml_problem == 'classification':
            corr = f_classif(dataframe[numerical_features],dataframe[target_features].to_numpy().ravel())
            for x in zip(dataframe[numerical_features].columns,corr[0],corr[1]):
                print(f"For {x[0]}: f statistic is {x[1]} and a p-value of {x[2]}")
    elif feature_selection_correlation == 'kendall':
        pass 
    elif feature_selection_correlation == 'chi-squared':
        pass
    



#%%
# fish = feature_selection(dataframe=df,
#                   feature_selection_correlation='ANOVA',
#                   type_of_ml_problem = 'classification',
#                   ordinal_features = dispatcher.features['ordinal_features'],
#                   nominal_features = dispatcher.features['nominal_features'],
#                   numerical_features = dispatcher.features['numerical_features'],
#                   target_features = dispatcher.features['target_feature'])

#%%
# #%%
# # define feature selection
# #fs = SelectKBest(score_func=f_classif, k=2)
# # apply feature selection
# #X_selected = fs.fit_transform(df[dispatcher.features['numerical_features']], df['churn'].to_numpy().ravel())
# #print(fs)
# # Create and fit selector


# selector = SelectKBest(f_classif, k=1)
# selector.fit(df[dispatcher.features['numerical_features']], 
#              df[dispatcher.features['target_feature']].to_numpy().ravel())
# # Get columns to keep and create new dataframe with those only
# cols = selector.get_support(indices=True)
# features_df_new = df.iloc[:,cols]

#%%
import pandas as pd
import numpy as np
#%%
def my_function(**kwargs):
    print(str(kwargs))

my_function(a=12, b="abc")