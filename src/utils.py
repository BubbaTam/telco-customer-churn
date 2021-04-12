# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:04:49 2021

@author: james
"""
import matplotlib.pyplot as plt




class evaluation:

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
        tp = evaluation.true_positive(y_true, y_pred,positive_class)
        fp = evaluation.false_positive(y_true, y_pred,positive_class)
        tn = evaluation.true_negative(y_true, y_pred,positive_class)
        fn = evaluation.false_negative(y_true, y_pred,positive_class)
        accuracy = (tp+tn) / (tp+fp+tn+fn)
        return accuracy
    def precision(y_true,y_pred,positive_class):
        """
        """
        tp = evaluation.true_positive(y_true, y_pred,positive_class)
        fp = evaluation.false_positive(y_true, y_pred,positive_class)
        precision = tp / (tp+fp)
        return precision
    def recall(y_true,y_pred,positive_class):
        """
        """
        tp = evaluation.true_positive(y_true, y_pred,positive_class)
        fn = evaluation.false_negative(y_true, y_pred,positive_class)
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
            p = evaluation.precision(y_true,prediction,positive_class)
            r = evaluation.recall(y_true,prediction,positive_class)
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
        p = evaluation.precision(y_true,y_pred,positive_class)
        r = evaluation.recall(y_true,y_pred,positive_class)
        f1_score = (2*p*r) / (p+r)
        return f1_score
    def tpr(y_true,y_pred,positive_class):
       """
       the same as recall 
       """
       tpr = evaluation.recall(y_true,y_pred,positive_class)
       return tpr 
    def fpr(y_true,y_pred,positive_class):
        """
        """
        fp = evaluation.false_positive(y_true, y_pred,positive_class)
        tn = evaluation.true_negative(y_true, y_pred,positive_class)
        fpr = fp / (fp+tn)
        return fpr