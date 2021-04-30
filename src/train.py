# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:04:33 2021

@author: james
"""
#%%
# The aim is to call all the relevant modules from this
#%%
#import libraries 
import os
import joblib
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
#from sklearn import metrics 
#from sklearn import tree 
import argparse

import src.config as config
import src.dispatcher as dispatcher
#import src.metrics
#import src.dataset as category 
import src.utils

#%%
b_f = ['gender','partner','dependents','phone_service',
                   'paperless_billing']
n_f = ['multiple_lines', 'internet_service','online_security',
                    'online_backup','device_protection','tech_support',
                    'streaming_tv','streaming_movies','contract',
                    'payment_method'
                    ]
target_feature = 'churn'
features_not_interest = ['churn','k_fold']
#%%
df = pd.read_csv('data/telco-customer-churn-clean-folds.csv')
#%%
def run(fold,
        features_not_interest,
        target_feature,
        binary_features,
        tune_model):
    """
    using split data, encoded the data, predicted using a model and gotter
    an evaluatin score
    """
    
    
    
    t_f = target_feature
    
    
    #read in the data
    df = pd.read_csv(config.TRAINING_FILE)
    
    
    
    # Encode the target feature
    df.loc[:,t_f] = LabelEncoder().fit_transform(df[t_f].values)
    
    
    # a list of the useful features for model 
    features = [
        f for f in df.columns if f not in features_not_interest]
    
    #separate into folds 
    # using the folds made earlier to create the training and validation set 
    # a shallow copy from the main dataframe
    df_train = df[df.k_fold != fold].reset_index(drop=True)
    df_valid = df[df.k_fold == fold].reset_index(drop=True)
    
    
    # prep the transformation for converting to numerical values
    t_features = [('binary',OrdinalEncoder(dtype='int'),b_f),
     ('nominal',OneHotEncoder(categories='auto',drop=None,sparse=False,dtype=int),n_f)]
    transformer_features = ColumnTransformer(t_features, remainder='passthrough')
    
    
    
   
    # fit the transformation to the combined training and validation
    # done on the index axis (placed below)
    # to deal the churn feature not having been sorted 
    data = pd.concat([df_train, df_valid],
                     axis=0
                     )
    # if churn already sorted
    # data = pd.concat([df_train[features], df_valid[features]],
    #                  axis=0
    #                  )
    transformer_features.fit(data[features])    
    #transform training data
    x_train = transformer_features.transform(df_train[features])
    
    #transform testing data
    x_valid = transformer_features.transform(df_valid[features])
    
    clf = dispatcher.models['logistic_regression']
    clf.fit(x_train,df_train['churn'])
    
    grid_search = GridSearchCV(estimator=clf,
                               param_grid=dispatcher.parameters['logistic_regression'],
                               n_jobs=-1)
    grid_search.fit(x_train,df_train[t_f].values)
    #print(f"")
    print(f"{grid_search.best_score_} : {grid_search.best_params_}")
            
    prediction = clf.predict(x_valid)
    print(prediction)
    #clf.score()
    print(src.utils.Evaluation.precision(df_valid['churn'],prediction))
    
#%%
    # using grid search for hyper parameter tuning 
    # want the best model to be signalled 
    for model in dispatcher.models['logistic_regression']:
        clf = dispatcher.models[model]
        if tune_model == 'grid search cv' and dispatcher.models == 'logistic_regression':
            grid_search = GridSearchCV(estimator=clf,
                                       param_grid=dispatcher.parameters['logistic_regression'],
                                       n_jobs=-1)
            grid_search.fit(x_train,df_train[t_f].values)
            print(f"{grid_search.best_score_} : {grid_search.best_params_}")
            grid_prediction = grid_search.predict(x_valid)
            report = classification_report(x_valid,grid_prediction)
            print(report)
            return 'fish'
            #auc = metrics.roc_auc_score(df_valid[t_f].values,prediction)
            #print(f"For model {model}\nfold {fold} has an auc score of {auc}")
            #print(prediction)
        elif tune_model == 'grid search cv' and dispatcher.models == 'XGBoost_classifier':
            grid_search = GridSearchCV(estimator=clf,
                                       param_grid=dispatcher.parameters['XGBoost_classifier'],
                                       n_jobs=-1)
            grid_search.fit(x_train,df_train[t_f].values)
            print(f"{grid_search.best_score_} : {grid_search.best_params_}")
            grid_prediction = grid_search.predict(x_valid)
        elif tune_model == 'grid search cv' and dispatcher.models == 'rf':
            grid_search = GridSearchCV(estimator=clf,
                                       param_grid=dispatcher.parameters['rf'],
                                       n_jobs=-1)
            grid_search.fit(x_train,df_train[t_f].values)
            grid_prediction = grid_search.predict(x_valid)
            print(f"{grid_search.best_score_} : {grid_search.best_params_}")
        elif tune_model == 'grid search cv' and dispatcher.models == 'decision_tree':
            grid_search = GridSearchCV(estimator=clf,
                                       param_grid=dispatcher.parameters['decision_tree'],  
                                       n_jobs=-1)
        
            grid_search.fit(x_train,df_train[t_f].values)
            grid_prediction = grid_search.predict(x_valid)
            print(f"{grid_search.best_score_} : {grid_search.best_params_}")
       # elif tune_model
        
        
        
        
        
        #clf.fit(x_train,df_train[t_f].values)
    
    #evaluation 
    # the probability of being '1'
    ####prediction = clf.predict_proba(x_valid)[:,1]
    prediction = clf.predict(x_valid)
    auc = metrics.roc_auc_score(df_valid[t_f].values,prediction)
    print(f"For model {model}\nfold {fold} has an auc score of {auc}")
    #######print(prediction)
    #accuracy = metrics.accuracy_score(y_valid,prediction)
    #print(f"fold = [fold], accuracy = {accuracy}")
    #saving the model 
    #joblib.dump(clf,f"/dt_{fold}.bin")
    
    # we want to create folders for each model -- that is whay we have 'model/'
    # config.MODEL_
    #####joblib.dump(clf,os.path.join(config.MODEL_OUTPUT,f"{model}",f"{model}_dt_{fold}.bin"))

#%%
for _ in range(5):
    run(fold=_,
        features_not_interest = features_not_interest,
        binary_features=b_f,
        target_feature=target_feature,
        tune_model='grid search cv')
#%%
# to loop through k_folds
for _ in range(5):
    run(fold=_,
        model='logistic_regression',
        features_not_interest = features_not_interest,
        target_feature = target_feature)
    
#%%
# philosophy 2   
run_2():
    """
    Information before:
        - we have no missing data 
        - 
    """
    # query data
    df = pd.read_csv(config.TRAINING_FILE)
    #subset = df[:1000]
    
    # data preparation 
    ##missing data
    ## clean data
    ## data transformation

















#%%
print(df[target_feature])
#%%
t_features = [('binary',OrdinalEncoder(dtype='int'),b_f)]
transformer_features = ColumnTransformer(t_features, remainder='passthrough')
dog = transformer_features.fit_transform(df) 
#print(transformer_features.get_feature_names())
#%%
#read in the data
df = pd.read_csv(config.TRAINING_FILE)
    # changes the target feature to 1 and 0
df.loc[:,target_feature] = LabelEncoder().fit_transform(df[target_feature].values)
    # geting a list of features that ignore k_fold and churn features 
features = [
    f for f in df.columns if f not in features_not_interest]
    # prep the transformation for converting to numerical values
t_features = [('binary',OrdinalEncoder(dtype='int'),b_f),
              ('nominal',OneHotEncoder(categories='auto',drop=None,sparse=False,dtype=int),n_f)]
transformer_features = ColumnTransformer(t_features, remainder='passthrough')
df_transformed = transformer_features.fit_transform(df)
df_train = df[df_transformed.k_fold != fold].reset_index(drop=True)
df_valid = df[df_transformed.k_fold == fold].reset_index(drop=True)
#%%
if __name__ == '__main__':
    #create a  parser that holds all our arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
        )
    parser.add_argument(
        "--model",
        type=str
        )
    args = parser.parse_args()
    run(fold=args.fold,model=args.model)

#%%
print(df.columns)
#%%
if __name__ == '__main__':
    run(fold=args.fold,model=args.model)
#%%
t_f = target_feature
    
    
    #read in the data
df = pd.read_csv(config.TRAINING_FILE)
    
    
    
    # Encode the target feature
df.loc[:,t_f] = LabelEncoder().fit_transform(df[t_f].values)
    
    
    # a list of the useful features for model 
features = [
    f for f in df.columns if f not in features_not_interest]
    
    #separate into folds 
    # using the folds made earlier to create the training and validation set 
    # a shallow copy from the main dataframe
df_train = df[df.k_fold != 0].reset_index(drop=True)
df_valid = df[df.k_fold == 0].reset_index(drop=True)
    
    
    # prep the transformation for converting to numerical values
t_features = [('binary',OrdinalEncoder(dtype='int'),b_f),
              ('nominal',OneHotEncoder(categories='auto',drop=None,sparse=False,dtype=int),n_f)]
transformer_features = ColumnTransformer(t_features, remainder='passthrough')
    
    
    
   
    # fit the transformation to the combined training and validation
    # done on the index axis (placed below)
    # to deal the churn feature not having been sorted 
data = pd.concat([df_train, df_valid],
                 axis=0
                 )
    # if churn already sorted
    # data = pd.concat([df_train[features], df_valid[features]],
    #                  axis=0
    #                  )
transformer_features.fit(data[features])    
    #transform training data
x_train = transformer_features.transform(df_train[features])
    
    #transform testing data
x_valid = transformer_features.transform(df_valid[features])
    
    
    # using grid search for hyper parameter tuning 
    # want the best model to be signalled 

clf = dispatcher.models['logistic_regression']
clf.fit(x_train,df_train['churn'])
clf.predict(x_valid)
#clf.score()
print(src.utils.Evaluation.roc_curve(df_valid['churn'],clf.predict(x_valid),positive_class = 1))



#parameters = {
        #'penalty':('l1', 'l2','elasticnet','none'),
        #'C':[0.01,0.1,1.0,10,100]}
#grid_search = GridSearchCV(estimator=clf,
         #                  param_grid=parameters,
          #                 n_jobs=-1)
#grid_search.fit(x_train,df_train[t_f].values)
##print(f"{grid_search.best_score_} : {grid_search.best_params_}")
#grid_prediction = grid_search.predict(x_valid)
#report = classification_report(x_valid[t_f],grid_prediction)
    #auc = metrics.roc_auc_score(df_valid[t_f].values,prediction)
    #print(f"For model {model}\nfold {0}")
#print(report)
#%%
print(dispatcher.models['logistic_regression'].get_params().keys())

'solver':['newton-cg','lbfgs','liblinear'],

#%%
print(df['k_fold'].unique())