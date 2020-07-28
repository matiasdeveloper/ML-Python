import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

#Import data
ds = pd.read_csv(r'C:\Users\matia\Desktop\GitHub\Machine-Learning-Standford-Course\Dataset CSV\diabetes.csv')
ds.head()

X = ds.iloc[:,0:7] # Features
X_test = X

print(ds.iloc[1,0:7])
y = np.array(ds.iloc[:,8]).reshape(-1,1) # Target variable
m = len(X)

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#Functions
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#Execute Logistic Regression Model
reg = LogisticRegression(max_iter=10000)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("LR classification predictions: ", y_pred)
print("\n")
print("LR classification score: ", score) 
print("\n")
print("LR classification accuracy: ", accuracy(y_test, y_pred))
print("\n")
print("The person with: ")
print("Pregnant = 3" + "\nGlucose = 75" + "\nBp = 55" + "\nSkin = 20" + "\nInsulin = 0" + "\nBmi = 22.5" + "\nPedigre = 0.9" + "\nAge = 35")

pacient = np.array([3,75,55,0,0,22.5,0.9]).reshape(-1,1)
print(pacient)

scorePacient1 = reg.predict(pacient)
print("Is " + scorePacient1)

#Plot the result