# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:25:20 2020

@author: Santosh Sah
"""
from PolynomialRegressionUtils import importPolynomialRegressionDataset, saveDataSetInPickle

def preprocess():
    
    X, y = importPolynomialRegressionDataset("Polynomial_Regression_Position_Salaries.csv")
    saveDataSetInPickle(X, y)
    

if __name__ == "__main__":
    preprocess()
