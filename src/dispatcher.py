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
models = {
    "decision_tree" : tree.DecisionTreeClassifier(),
    "rf" : ensemble.RandomForestClassifier(),
    "logistic_regression" : linear_model.LogisticRegression(max_iter=400),
    "XGBoost_classifier" : xgb.XGBClassifier(),
    "SVM": svm.SVC()
    }
#%%
parameters = {
    'XGBoost_classifier' :
        {
        'eta':[0.01,0.015,0.025,0.05,0.1],
        'gamma':(0.05,0.1,0.3,0.5,0.7,0.9,1.0),
        'max_depth':[3,5,7,9,12,15,17,25],
        'min_child_weight': [1,3,5,7],
        'subsample': [0.6,0.7,0.8,0.9,1.0],
        'colsample_bytree': [0.6,0.7,0.8,0.9,1.0],
        'lambda': [0.01,0.1,1.0],
        'alpha': [0,0.1,0.5,1.0]
        },
    'logistic_regression' :
        {
        'C':[0.01,0.05,0.1,0.5,1.0,10]
        },
    'rf' :
        {
        'n_estimators':[120,300,500,800,1200],
        'max_depth':[5,8,15,25,30,None],
        'min_samples_split': [1,2,5,10,15,100],
        'min_samples_leaf': [1,2,5,10],
        'max features': ['auto','log2','sqrt',None]
        },
    'decision_tree' :
        {
            'criterion': ['gini','entropy'],
            'max_depth':[5,8,15,25,30,None],
            'min_samples_split': [1,2,5,10,15,100],
            'min_samples_leaf': [1,2,5,10],
            'max features': ['auto','log2','sqrt',None]
        },
    'SVC' :
        {
        'C':[0.001,0.01,0.1,10,100,1000],
        'gamma':['gamma','auto'],
        'class_weight': ['balanced',None]}
                }
ordinal_feature_order = None
#%%
print(matplotlib.__version__)
# {
#         'solver':['newton-cg','lbfgs','liblinear'],
#         'penalty':('l1', 'l2','elasticnet','none'),
#         'C':[0.01,0.1,1.0,10,100]
#         },
