# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:27:31 2020

@author: Santosh Sah
"""

import matplotlib.pyplot as plt
import numpy as np
from PolynomialRegressionUtils import (readLinearRegressionModel, readIndepentDataset, readDependentDataset, readPolynomialRegressionModel,
                                       readPolynomialRegressionModelForVisualization)
"""
Visualizing training set results for linear regression
"""
def visualisingTrainingSetResult():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    linearRegressionModel = readLinearRegressionModel()
    
    # Visualising the Linear Regression results
    plt.scatter(X, y, color = "red")
    plt.plot(X, linearRegressionModel.predict(X), color = "blue")
    plt.title("Truth or Bluff (Linear Regression)")
    plt.xlabel("Position level")
    plt.ylabel("Salary")
    
    plt.savefig("linear_regression_trainingsetresult.png")
    
    plt.show()

"""
Visualizing training set results for polynomial regression

"""
def visualisingTrainingSetResultForPolynomialRegression():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    polynomialRegressionModel = readPolynomialRegressionModel()
    polynomialRegressionModelForVisualization = readPolynomialRegressionModelForVisualization()

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, polynomialRegressionModel.predict(polynomialRegressionModelForVisualization.fit_transform(X)), color = 'blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    
    plt.savefig("polynomial_regression_trainingsetresult.png")
    
    plt.show()

"""
Visualising the Polynomial Regression results (for higher resolution and smoother curve)

"""
def visualisingPolynomialRegressionInHighResolution():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    polynomialRegressionModel = readPolynomialRegressionModel()
    polynomialRegressionModelForVisualization = readPolynomialRegressionModelForVisualization()
    
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))    

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, polynomialRegressionModel.predict(polynomialRegressionModelForVisualization.fit_transform(X_grid)), color = 'blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    
    plt.savefig("polynomial_regression_trainingsetresult_high_resolution.png")
    
    plt.show()
    
if __name__ == "__main__":
    #visualisingTrainingSetResult()
    #visualisingTrainingSetResultForPolynomialRegression()
    visualisingPolynomialRegressionInHighResolution()