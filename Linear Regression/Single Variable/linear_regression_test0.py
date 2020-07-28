import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from linear_regression import LinearRegression

ds = pd.read_csv(r'C:\Users\matia\Desktop\GitHub\Machine-Learning-Standford-Course\Dataset CSV\ex1data1.csv')
print(ds)

X = ds.iloc[:,0]
y = ds.iloc[:,1]
n = len(X)

#X, y = datasets.make_regression(n_samples = 100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2, random_state = 1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X, y, color="b", marker="o", s = 30)
plt.show()

reg = LinearRegression(lr=0.01)
reg.fit(X_train, y_train, n)
predicted = reg.predict(X_test)

def mse(y_true, y_predicted):
    return np.mean(y_true - y_predicted**2)

mse_value = mse(y_test, predicted)
print(mse_value)

y_predic_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1= plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m1= plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X,y_predic_line, color='black', linewidth=2, label='prediction')
plt.show()
