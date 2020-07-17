import numpy as np 
from sklearn import linear_model 

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1500, normalize = False, intercept = 0, coef = 0):
        self.lr = lr
        self.n_iters = n_iters
        self.normalize = normalize
        self.regr = linear_model
        self.intercept = intercept
        self.coeficient = coef
    
    def fit(self, X_train, y_train):
        n_samples = len(X_train)
        n_features = len(X_train)

        self.regr = linear_model.Ridge(alpha=self.lr, normalize=self.normalize, max_iter=self.n_iters)
        self.regr.fit(X_train, y_train)
        self.intercept = self.regr.intercept_
        self.coeficient = self.regr.coef_


    def predict(self, X_test):
        predicted = self.regr.predict(X_test)        
        return predicted

    def interceptCoeficient(self):
        return self.coeficient, self.intercept

