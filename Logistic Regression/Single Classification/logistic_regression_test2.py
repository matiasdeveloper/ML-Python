import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

#Import data
ds = pd.read_csv(r'C:\Users\matia\Desktop\GitHub\Machine-Learning-Standford-Course\Dataset CSV\insurance_data.csv')

X = ds.age

y = ds.bought_insurance

print(ds)

X_train, X_test, y_train, y_test = train_test_split(ds[['age']],ds.bought_insurance, test_size=0.1)
# Normalize features
#mms = MinMaxScaler()
#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.transform(X_test)

#Plot data with matplot
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color="b", marker="o", s = 20)
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')
plt.show()

#Functions
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#Execute Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions2 = model.predict(np.array(20).reshape(-1,1))

print(predictions)
print(predictions2)

#Plot the result