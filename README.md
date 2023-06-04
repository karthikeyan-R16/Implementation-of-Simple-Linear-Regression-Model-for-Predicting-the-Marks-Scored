# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe
4. Plot the required graph both for test data and training data.

## program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Karthikeyan R
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

## output:

![ml2](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/7051beec-6b00-4f5d-9db3-10efb82a347b)

![ml2 1](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/8e3e1c42-3652-4d9a-b68c-ffb328a4ab75)

![ml2 2](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/c93068a9-8dab-4355-8823-8f5f7a5981fa)

![ml2 3](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/22c98f06-a6f4-4491-8fc7-1c5c11a1c96b)

![ml2 4](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/81791894-e3e3-4905-a795-f1d2ec8c902c)

![ml2 5](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/5338df86-f35f-4c13-8afd-e5bf56346ad9)

![ml2 6](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/8b70213f-1969-47a8-8468-ff34dc295867)

![ml2 7](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/2d774d80-4cf0-4b89-a80c-a9fa87c4cf32)


![ml2 8](https://github.com/karthikeyan-R16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119421232/5e2e9960-2e1e-4a2e-984c-4764f447731b)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
