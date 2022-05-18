![hour](https://user-images.githubusercontent.com/105532515/169054053-6cee2974-74ea-4645-be64-8d90e6bccdd8.png)
![hour](https://user-images.githubusercontent.com/105532515/169054168-3e515bdb-566b-49f3-9cae-e7fc791eef75.png)
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ganesh p
RegisterNumber:  212220040112
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
X
Y = dataset.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="brown")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="brown") 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/

## OUTPUT:
![hour](https://user-images.githubusercontent.com/105532515/169054628-ed35e81e-6353-4cea-a05f-ee676f150aac.png)
![minute (1)](https://user-images.githubusercontent.com/105532515/169055624-80ea49fb-2bb3-4f9b-80d5-ce781df5a001.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.



