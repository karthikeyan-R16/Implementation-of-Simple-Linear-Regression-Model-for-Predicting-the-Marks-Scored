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
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error 
df=pd.read_csv('/student_scores.csv')
df.head()

x = df.iloc[:,:-1].values
x
#segregating data to variables

y= df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

y_pred
y_test

plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="brown")
plt.title("Hours Vs Scores(Training set")
plt.xlabel ("Hours")
plt.ylabel ("Scores")
plt.show()

plt.scatter(x_test,y_test,color="black")
plt.plot(x_test,reg.predict(x_test),color="magenta")
plt.title("Hours Vs Scores(Test set")
plt.xlabel ("Hours")
plt.ylabel ("Scores")
plt.show()

mse = mean_squared_error(y_test,y_pred)
print("MSE = ",mse)

mae = mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)

## Output:
![output](./img1.png)

![output](./img%202.png)

![output](./img3.png)

![output](./img%204.png)

![output](./img5.png)

![output](./img6.png)

![output](./img%207.png)

![output](./img%208.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
