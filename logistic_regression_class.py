""""
Machine Learning - Logistic Regression with Gradient Descent Algorithm
version: 0.1
author: Ade Kurniawan
description: this is my personal project to try to build Gradient Descent algorithm from the ground up. If you learn machine learning and data science or even if you are a data scientist yourself, you would have no difficulties in understanding this code. The symbols and terms used here are influenced by the ones used in the Andrew Ng's Machine Learning course from Coursera. The structure of the program is inspired from the structure of Scikit-learn. This code is basically not challenging at all if you have tried to build a gradient descent algorithm for linear regression.
for information about Logistic Regression, visit this link https://en.wikipedia.org/wiki/Logistic_regression
"""

import numpy as np

#sigmoid function
def sigmoid(z):
        return 1/(1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, init_weights = 'default', alpha = 0.01, iterations = 1500, C = 0):
        self.w = init_weights # weights
        self.alpha = alpha # learning rate
        self.n_iter = iterations # number of iterations which will be performed
        self.C = C # regularization term
    def fit(self, X, y):
        y = y.reshape(-1,1)
        m, n = X.shape # number of datapoints, number of features/variables
        if self.w == 'default':
            #if default,initial weights are set to zero
            self.w = np.zeros((n+1, 1)) # +1 to include the intercept part of the model
        elif self.w =='random':
            #if random, the initial weights are set using random number generator, near zero value to avoid divergence
            self.w = np.random.normal(0,0.5, (n+1, 1))
        # add one column containing value 1s to  X to accomodate intercept, 
        ones = np.ones((m,1))
        X = np.concatenate((ones,X), axis = 1)
        #everthing is ready, commencing the iterations
        delta = np.concatenate((np.zeros((1,1)), np.ones((n,1))), axis = 0) #simply to exclude w[0] from being regularized (it's a matter of convention as far as I can remember)
        for k in range(self.n_iter):
            h = sigmoid(X @ self.w)
            grad = np.dot(X.T, h-y).sum(axis=1).reshape(-1, 1)
            self.w = self.w - self.alpha*grad/m - self.C*self.w*delta/m
    def predict(self, X):
        X = X.reshape(-1,1)
        prediction = sigmoid(self.w[0] + self.w[1:].T @ X)
        return prediction 
    def get_params(self):
        return self.w
# testing the program
# unfinished, it has been tested yet. I'm still figuring out the best way to simulate a prediction using this code
