# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:04:33 2021

@author: james
"""
#%%
#import libraries 
import joblib
import pandas as pd 
from sklearn import metrics 
from sklearn import tree 

#%%
def run(fold):
    df = pd.read_csv("data/telco-customer-churn-clean-folds.csv")
    df_train = df[df.k_fold != fold].reset_index(drop=True)
    df_valid = df[df.k_fold == fold].reset_index(drop=True)
    x_train = df_train.drop('churn',axis=1).values
    y_train = df_train.churn.values
    x_valid = df_valid.drop('churn',axis=1).values
    y_valid = df_valid.churn.values
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    prediction = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid,prediction)
    print(f"fold = [fold], accuracy = {accuracy}")
    joblib.dump(clf,f"models/dt_{fold}.bin")

if __name__ == "__main__":
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
#%%
import os
print(os.getcwd())
df = pd.read_csv("data/telco-customer-churn-clean-folds.csv")