import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn import linear_model 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from linear_regressionmulti import LinearRegression

ds = pd.read_csv(r'C:\Users\matia\Desktop\ex1data2.csv')
print(ds)

X = ds.iloc[:,0:1]
y = ds.iloc[:,2]
n = len(X)

# Build the model with Cost Function
X = X.values.reshape(-1,1)
y = y.values.reshape(-1,1)

#X, y = datasets.make_regression(n_samples = 100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.8, random_state = 1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X, y, color="b", marker="o", s = 100)
plt.show()

# with sklearn
regr = LinearRegression(0.001, 1500, True)
regr.fit(X_train, y_train)
predicted = regr.predict(X_test)
coef, intercept = regr.interceptCoeficient()

print('Intercept: \n', intercept)
print('Coefficients: \n', coef)

y_pred = predicted
compare=pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(compare)

fig, (ax1) = plt.subplots(1,figsize =(6,5))
ax1.scatter (X_test, y_test, s = 20)
plt.plot(X_test, y_pred, color = 'black', linewidth = 2)
plt.show()


