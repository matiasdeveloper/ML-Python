import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        #self.theta = None
    
    def fit(self, X, y):
        #Init paramtetes
        n_samples, n_features = len(X)
        self.weights = np.zeros(n_features)
        self.bias = 0
        #self.theta = init_theta(X)

        #Gradient descent
        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            print(y_predicted)
            #theta = self.weights + self.lr * gradient
    
    def predict(self, X):
            linear_model = np.dot(X, self.weights) + self.bias

            y_predicted = self._sigmoid(linear_model)
            y_predicted_cls = [1 if i >= 0.5 else 0 for i in y_predicted]
            
            return y_predicted_cls
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))