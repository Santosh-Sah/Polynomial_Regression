# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:26:53 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Import dataset and read specific column. Split the dataset in training and testing set.
Data set is very small and hence we are not going to divide the dataset in training and test set.
We will train our model on the whole dataset
"""
def importPolynomialRegressionDataset(linearRegressionDatasetFileName):
    
    linearRegressionDataset = pd.read_csv(linearRegressionDatasetFileName)
    X = linearRegressionDataset.iloc[:, 1:2].values
    y = linearRegressionDataset.iloc[:, 2].values
    
    """
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test
    
    """
    
    return X, y

"""
Save dataset as pickle file
"""
def saveDataSetInPickle(X, y):
    
    #Write X in a picke file
    with open("X.pkl",'wb') as X_Pickle:
        pickle.dump(X, X_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("y.pkl",'wb') as y_Pickle:
        pickle.dump(y, y_Pickle, protocol = 2)

"""
Save PolynomialLinearRegressionModel as a pickle file.
"""
def savePolynomialRegressionModel(PolynomialRegressionModel):
    
    #Write PolynomialRegressionModel as a picke file
    with open("PolynomialRegressionModel.pkl",'wb') as PolynomialRegressionModel_Pickle:
        pickle.dump(PolynomialRegressionModel, PolynomialRegressionModel_Pickle, protocol = 2)


"""
read polynomialRegressionModel from pickle file
"""
def readPolynomialRegressionModel():
    
    #load PolynomialRegressionModel model
    with open("PolynomialRegressionModel.pkl","rb") as PolynomialRegressionModel:
        polynomialRegressionModel = pickle.load(PolynomialRegressionModel)
    
    return polynomialRegressionModel

"""
read X from pickle file
"""
def readIndepentDataset():
    
    #load y_test
    with open("X.pkl","rb") as X_pickle:
        X = pickle.load(X_pickle)
    
    return X

"""
read y from pickle file
"""
def readDependentDataset():
    
    #load y
    with open("y.pkl","rb") as y_pickle:
        y = pickle.load(y_pickle)
    
    return y

"""
Save LinearRegressionModel as a pickle file.
"""
def saveLinearRegressionModel(LinearRegressionModel):
    
    #Write LinearRegressionModel as a picke file
    with open("LinearRegressionModel.pkl",'wb') as LinearRegressionModel_Pickle:
        pickle.dump(LinearRegressionModel, LinearRegressionModel_Pickle, protocol = 2)


"""
read LinearRegressionModel from pickle file
"""
def readLinearRegressionModel():
    
    #load LinearRegressionModel model
    with open("LinearRegressionModel.pkl","rb") as LinearRegressionModel:
        linearRegressionModel = pickle.load(LinearRegressionModel)
    
    return linearRegressionModel

"""
Save PolynomialLinearRegressionModel as a pickle file.
"""
def savePolynomialRegressionModelForVisualization(PolynomialRegressionModelForVisualization):
    
    #Write PolynomialRegressionModel for visualization as a picke file
    with open("PolynomialRegressionModelForVisualization.pkl",'wb') as PolynomialRegressionModelForVisualization_Pickle:
        pickle.dump(PolynomialRegressionModelForVisualization, PolynomialRegressionModelForVisualization_Pickle, protocol = 2)


"""
read polynomialRegressionModel for visualization from pickle file
"""
def readPolynomialRegressionModelForVisualization():
    
    #load PolynomialRegressionModel model
    with open("PolynomialRegressionModelForVisualization.pkl","rb") as PolynomialRegressionModelForVisualization:
        polynomialRegressionModelForVisualization = pickle.load(PolynomialRegressionModelForVisualization)
    
    return polynomialRegressionModelForVisualization