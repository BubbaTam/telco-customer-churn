# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:03:07 2021

@author: james
"""
# can add or remove models from here


#import library 
import sklearn
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm 
import xgboost as xgb
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
#%%
features = {
    'binary_features' : ['gender','senior_citizen','partner','dependents','phone_service',
                   'paperless_billing'],
    'nominal_features' : ['multiple_lines', 'internet_service','online_security',
                    'online_backup','device_protection','tech_support',
                    'streaming_tv','streaming_movies','contract',
                    'payment_method'
                    ],
    'ordinal_features' : None,
    'numerical_features' : ['tenure','monthly_charges','total_charges'],
    'target_feature' : ['churn'],
    'fold_feature' : ['k_fold']
    }

#%%
models = {
    "decision_tree" : tree.DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_split=100,min_samples_leaf=10),
    "rf" : ensemble.RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_split=100,min_samples_leaf=5),
    "logistic_regression" : linear_model.LogisticRegression(max_iter=400,C=0.01),
    "XGBoost_classifier" : xgb.XGBClassifier(),
    "SVM": svm.SVC()
    }
#%%
parameters = {
    'XGBoost_classifier' :
        {
        'eta':[0.01,0.015],
        'gamma':[0.05,0.1],
        'max_depth':[3,5],
        'min_child_weight': [1,3],
        'subsample': [0.6,0.7],
        'colsample_bytree': [0.6,0.7],
        'lambda': [0.01,0.1],
        'alpha': [0,0.1,0]
        },
    'logistic_regression' :                    
        {
        'C':[0.01,0.05,0.1,0.5,1.0,10]
        },
    # possible parameters for logistic regression with sklearn:
        # penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
        # C: float, default=1.0
        
    'rf' : # fine
        {
        'n_estimators':[120,300,500,800,1200],
        'max_depth':[5,8,15,25,30],
        'min_samples_split': [2,5,10,15,100],  
        'min_samples_leaf': [1,2,5,10]
        },
    'decision_tree' :
        {
        'criterion': ['gini','entropy'],
        'max_depth':[5,8,15,25,30],
        'min_samples_split': [2,5,10,15,100],
        'min_samples_leaf': [1,2,5,10]
        },
    'SVM' :
        {
        'kernel':['poly','rbf','sigmoid'],
        'C':[0.01,0.1,1,10,100,1000],
        'gamma':[0.001,0.01,0.1,1]
        }
                }