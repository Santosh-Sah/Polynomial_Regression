# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:24:42 2020

@author: Santosh Sah
"""

import pandas as pd
from PolynomialRegressionUtils import readLinearRegressionModel, readPolynomialRegressionModel, readPolynomialRegressionModelForVisualization

def predictLinearRegression():
    
    linearRegression = readLinearRegressionModel()
    
    inputValue = [6.5]
    inputValueDataframe = pd.DataFrame(inputValue)
    
    predictedValue = linearRegression.predict(inputValueDataframe.values)
    
    print(predictedValue)

def predictPolynomialRegression():
    
    polynomialRegressionModel = readPolynomialRegressionModel()
    polynomialRegressionModelForVisualization = readPolynomialRegressionModelForVisualization()
    
    inputValue = [6.5]
    inputValueDataframe = pd.DataFrame(inputValue)
    
    predictedValue = polynomialRegressionModel.predict(polynomialRegressionModelForVisualization.fit_transform(inputValueDataframe.values))
    
    print(predictedValue)


if __name__ == "__main__":
    #predictLinearRegression()
    predictPolynomialRegression()

