# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:04:49 2021

@author: james
"""





class evaluation:

    def accuracy(y_true, y_pred):
        """
        Returns the accuracy score of two arrays 
        """
        # Firstly make a counter variable and set to 0.
        accuracy_counter = 0
        # Use a zip statement to combine the predicted and true value array
        for (y_t, y_p) in zip(y_true,y_pred):                                     
            if y_t == y_p:                                                         
                accuracy_counter +=1                                              
                return accuracy_counter / len(y_true)     
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

        Parameters
        ----------
        y_true : TYPE
            DESCRIPTION.
        y_pred : TYPE
            DESCRIPTION.
        positive_class : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """                 
        true_negative_counter = 0
        for yt,yp in zip(y_true,y_pred):
            if yt != positive_class and yp != positive_class:
                true_negative_counter+=1
        return true_negative_counter  
    def false_negative(y_true, y_pred,positive_class)
