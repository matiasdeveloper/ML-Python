import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regressionmulti import LogisticRegressionMulti
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

#Import data
ds = pd.read_csv(r'C:\Users\matia\Desktop\GitHub\Machine-Learning-Standford-Course\Dataset CSV\ex2data1.csv')

X = ds.iloc[:,0:1]
X_test = X

y = ds.iloc[:,2]
m = len(X)

theta = np.zeros((X.shape[1], 1))

print(ds)

#Normalize features
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X)
X_test_norm = mms.transform(X_test)

#Plot data with matplot
fig = plt.figure(figsize=(8,6))
plt.scatter(X, y, color="b", marker="o", s = 50)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

#Functions
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#Execute Logistic Regression Model
reg = LogisticRegression(lr=0.1, n_iters=2000)
reg.fit(X_train_norm, y)
predictions = reg.predict(X_test_norm)

print("LR classification accuracy: ", accuracy(y, predictions))

#Plot the result