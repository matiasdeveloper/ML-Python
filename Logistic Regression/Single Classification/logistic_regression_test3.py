import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

#Import data
ds = pd.read_csv(r'C:\Users\matia\Desktop\GitHub\Machine-Learning-Standford-Course\Dataset CSV\logisticRegressionSubmit.csv')

X = ds.iloc[:,0]
y = ds.iloc[:,1]

print(ds)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)

# Normalize features
#mms = MinMaxScaler()
#X_train_norm = mms.fit_transform(np.array(X_train).reshape(-1,1))
#X_test_norm = mms.transform(np.array(X_test).reshape(-1,1))

#Plot data with matplot
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color="b", marker="s", s = 10)
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')
plt.show()

#Execute Logistic Regression Model
model = LogisticRegression(random_state=0, max_iter=400,tol=0.001)
model.fit(np.array(X_train).reshape(-1,1), y_train)

y_pred = model.predict(np.array(X_test).reshape(-1,1))
print(y_pred)

predictions2 = model.predict(np.array(100).reshape(-1,1))
print(predictions2)

score = accuracy_score(y_test, y_pred)
print(score)
#Plot the result