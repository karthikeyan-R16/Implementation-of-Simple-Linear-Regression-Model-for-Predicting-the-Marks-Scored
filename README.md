# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 212222240045
RegisterNumber: KARTHIKEYAN R
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X = df.iloc[:,:-1].values
X

y = df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

y_pred

y_test

plt.scatter(X_train,y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,y_test,color="grey")
plt.plot(X_test,regressor.predict(X_test),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/
## OUTPUT
![11](https://user-images.githubusercontent.com/119421232/232498547-60f23e51-aadd-41d4-b892-a7b3ace5a715.png)

![12](https://user-images.githubusercontent.com/119421232/232498639-eda73573-0162-444d-aa87-a00bb17c6f4f.png)

![13](https://user-images.githubusercontent.com/119421232/232498716-7e5001bf-75e2-4b49-b129-b102d2859999.png)

![14](https://user-images.githubusercontent.com/119421232/232498785-8efc999d-cfd8-4331-bd4a-358a6426956c.png)

![15](https://user-images.githubusercontent.com/119421232/232498870-187521cf-1ede-43e1-b082-69e036937b0d.png)

![16](https://user-images.githubusercontent.com/119421232/232498963-44ab65a5-7ce7-4580-9b8f-ef755247b278.png)

![17](https://user-images.githubusercontent.com/119421232/232500238-d1e1c28e-c34c-4023-8f65-fabf23892222.png)

![18](https://user-images.githubusercontent.com/119421232/232500298-0d76ade9-105f-46bb-888e-0870f90fd9e9.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
