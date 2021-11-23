# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:45:03 2021

@author: james
"""
print(2+2)
#%%
# plans to make an interactive categorical variable encoding 
# def deal_with_categorical_not_fin(df):
#     """
#     steps:
#         - identify categorical features 
#         - give information to the user about the feature 
#     """
#     categorical_variables = []
#     transformer_list = []
#     string = 0
#     try:
#         [categorical_variables.append(x) for x in df.columns if df[x].dtypes == object]
#     except:
#         print("there is an issue with the list comprehension")
#     for x in categorical_variables:
#         question = input(f"Is feature {x.upper()}:\nIt has {df[x].nunique()} features with the features of \n{df[x].unique()}? ")
#         # getting the right format of input 
#         question = question.lower()
#         if question == 'ordinal':
#             function = OrdinalEncoder()
#             #will need to put in own order system 
#             ordinal_question = input(f"Can you give the desired order of {x} ")
#                 #will be taking in a string of the input
#                 #we want to put this into a list 
#             order_1 = ordinal_question.split(', ')
#             print(order_1)
#             ordinal = OrdinalEncoder(categories=[order_1],dtype='int')
#             ordinal.fit(df[[x]])
#             df[x] = ordinal.transform(df[[x]])
#         elif question == 'nominal':
#             dropped_level = input(f"what level should be dropped from {df[x].unique()}")
#             function = OneHotEncoder(drop=[dropped_level],dtype=int,sparse=False)
#             ohe.fit(df[x])
#             ohe.transform(df[x])
#         elif question == 'stop':
#             break
#         else:
#             print("bye")
        
#         (str(x),function,)
    

