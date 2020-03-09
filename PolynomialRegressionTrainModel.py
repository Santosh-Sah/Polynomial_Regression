# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:26:05 2020

@author: Santosh Sah
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from PolynomialRegressionUtils import (savePolynomialRegressionModel, readIndepentDataset, readDependentDataset, saveLinearRegressionModel,
                                       savePolynomialRegressionModelForVisualization)

"""
Train polynomial regression model 
"""
def trainPolynomialRegressionModel():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    
    # Fitting Polynomial Regression to the dataset
    polynomialFeatures = PolynomialFeatures(degree=4)
    X_Polynomial = polynomialFeatures.fit_transform(X)
    
    polynomialFeatures.fit(X_Polynomial, y)
    savePolynomialRegressionModelForVisualization(polynomialFeatures)
    
    polynomialLinearRegression = LinearRegression()
    polynomialLinearRegression.fit(X_Polynomial, y)
    
    savePolynomialRegressionModel(polynomialLinearRegression)

"""
Train linear regression model 
"""
def trainLinearRegressionModel():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    
    # Fitting Linear Regression to the dataset
    linearRegression = LinearRegression()
    linearRegression.fit(X, y)
    
    saveLinearRegressionModel(linearRegression)

if __name__ == "__main__":
    #trainLinearRegressionModel() 
    trainPolynomialRegressionModel()
