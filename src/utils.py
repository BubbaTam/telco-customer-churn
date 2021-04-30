# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:04:49 2021

@author: james
"""
import matplotlib.pyplot as plt


class DataExploration:
    def __init__(self,*args):
        self.args = args
    def number_of_unique_values(dataframe):
        """
        """
        for column_name in dataframe.columns:
            if dataframe[column_name].dtype == object:
                if dataframe[column_name].nunique() >= 10:
                    print(f"Best to individually look at at {column_name.upper()}")
                else:
                    print(f"For {column_name.upper()} \n The values are:{dataframe[column_name].unique()}")
class FeatureEng:
    def __init__(self,*args):
        self.args = args
    

class Evaluation:
    def __init__(self,*args):
        self.args = args
    def true_positive(y_true, y_pred,positive_class):
        """
        """
        true_positive_counter = 0
        for yt,yp in zip(y_true,y_pred):
            if yt == positive_class and yp == positive_class:
                true_positive_counter+=1
        return true_positive_counter       
    def true_negative(y_true,y_pred,positive_class):
        """
        """                 
        true_negative_counter = 0
        for yt,yp in zip(y_true,y_pred):
            if yt != positive_class and yp != positive_class:
                true_negative_counter+=1
        return true_negative_counter  
    def false_negative(y_true, y_pred,positive_class):
        """
        """
        false_negative_counter = 0 
        for (y_t,y_p) in zip(y_true,y_pred):
            if y_p != positive_class and y_t == positive_class:
                false_negative_counter += 1
        return false_negative_counter
    def false_positive(y_true,y_pred,positive_class):
        """
        """
        false_positive_counter = 0 
        for (y_t,y_p) in zip(y_true,y_pred):
            if y_p == positive_class and y_t != positive_class:
                false_positive_counter +=1
        return false_positive_counter
    def accuracy(y_true, y_pred,positive_class):
        """
        Returns the accuracy score of two arrays:
            (TP+TN) / (TP+FP+TN+FN)
        """
        # Firstly make a counter variable and set to 0.
        tp = Evaluation.true_positive(y_true, y_pred,positive_class)
        fp = Evaluation.false_positive(y_true, y_pred,positive_class)
        tn = Evaluation.true_negative(y_true, y_pred,positive_class)
        fn = Evaluation.false_negative(y_true, y_pred,positive_class)
        accuracy = (tp+tn) / (tp+fp+tn+fn)
        return accuracy
    def precision(y_true,y_pred,positive_class=1):
        """
        """
        tp = Evaluation.true_positive(y_true, y_pred,positive_class)
        fp = Evaluation.false_positive(y_true, y_pred,positive_class)
        precision = tp / (tp+fp)
        return precision
    def recall(y_true,y_pred,positive_class):
        """
        """
        tp = Evaluation.true_positive(y_true, y_pred,positive_class)
        fn = Evaluation.false_negative(y_true, y_pred,positive_class)
        recall = tp / (tp +fn)
        return recall
    def precision_recall_curve_binary(y_true, y_pred,positive_class=1,thresholds=[0.1,
                                                                                  0.2,
                                                                                  0.3,
                                                                                  0.4,
                                                                                  0.5,
                                                                                  0.6,
                                                                                  0.7,
                                                                                  0.8]):
        """
        note to self:
            
        """
        precision = []
        recall=[]
        for i in thresholds:
            prediction = [1 if x >= i else 0 for x in y_pred]
            p = Evaluation.precision(y_true,prediction,positive_class)
            r = Evaluation.recall(y_true,prediction,positive_class)
            precision.append(p)
            recall.append(r)
            #graph 
            plt.figure(figsize=(20,15))
            plt.plot(recall,precision)
            plt.title(f'Threshold {i}',fontsize=15)
            plt.xlabel('Recall',fontsize=12)
            plt.ylabel('Precision',fontsize=12)
            plt.show()        
    def f1(y_true,y_pred,positive_class):
        """
        ways: 
            f1 = (2*precision*recall)/ (precision + recall)
            f1 = (2*True positive) / (2 * true positive + false positive + false negative)
        """
        p = Evaluation.precision(y_true,y_pred,positive_class)
        r = Evaluation.recall(y_true,y_pred,positive_class)
        f1_score = (2*p*r) / (p+r)
        return f1_score
    def tpr(y_true,y_pred,positive_class):
       """
       the same as recall 
       """
       tpr = Evaluation.recall(y_true,y_pred,positive_class)
       return tpr 
    def fpr(y_true,y_pred,positive_class):
        """
        """
        fp = Evaluation.false_positive(y_true, y_pred,positive_class)
        tn = Evaluation.true_negative(y_true, y_pred,positive_class)
        fpr = fp / (fp+tn)
        return fpr
    def true_negative_rate(y_true,y_pred,positive_class):
        """
        A.K.A 
        """
        tnr = 1 - Evaluation.fpr(y_true,y_pred,positive_class)
        return tnr
    def roc_curve(y_true, y_pred,positive_class=1,thresholds=[0.1,
                                                              0.2,
                                                              0.3,
                                                              0.4,
                                                              0.5,
                                                              0.6,
                                                              0.7,
                                                              0.8]):
        """
        - binary 
        purpose to test out different threshold values
        used with true positive rate (TPR) (tp / (tp+fn)) and 
        false positive rate (FPR) (FP / (FP+TN))
        """
        tpr_list = []
        fpr_list = []
        for thresh in thresholds:
            pred = [1 if x >= thresh else 0 for x in y_pred]
            tpr_temp = Evaluation.tpr(y_true,pred,positive_class)
            fpr_temp = Evaluation.fpr(y_true,pred,positive_class)
            tpr_list.append(tpr_temp)
            fpr_list.append(fpr_temp)
        plt.figure(figsize=(20,15))
        plt.fill_between(fpr_list,tpr_list)
        plt.plot(fpr_list,tpr_list)
        plt.xlim(0,1.0)
        plt.ylim(0,1.0)
        plt.xlabel('FPR',fontsize=12)
        plt.ylabel('TPR',fontsize=12)
        plt.show()
        