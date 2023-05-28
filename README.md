# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your cell. 
2. Type the required program
3. Print the program.
4. End the program.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHIKEYA R
RegisterNumber:212222240045

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
print(x)

y=df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output
![mk1](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/c0da7c5f-a4f1-4030-918e-cfea6d313043)

![mk2](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/5f098656-afd5-4887-abab-0861cc66e3cb)

![mk3](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/4db17357-70f5-4ee2-8970-2d4873cfb00c)

![mk4](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/50ba6d74-f516-43db-9a71-6e73c3deb589)

![mk5](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/89c415d0-3f05-4c8d-a49a-2a5f6645fd13)

![mk6](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/7e6c250a-dd3f-4d28-8469-928deb97d2d3)


![mk7](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/774e1a89-c3fe-4929-9051-f17bef89ae3f)

![mk8](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/2dd26187-0581-4fcf-90f0-3c6b89db04ed)

![mk9](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/28b20e5f-0137-47dc-99fd-f1a7f6f344d7)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
