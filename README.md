# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored


## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Step 1.Start
Step 2.Import the standard Libraries.
Step 3.Set variables for assigning dataset values.
Step 4.Import linear regression from sklearn.
Step 5.Assign the points for representing in the graph.
Step 6.Predict the regression for marks by using the representation of the graph.
Step 7.Compare the graphs and hence we obtained the linear regression for the given datas.
Step 8.End
```


## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Iswarya P
RegisterNumber:  212223230082

```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train.shape)
print(x_test.shape)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,Y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:
## Dataset value:
![image](https://github.com/user-attachments/assets/b795bc34-7754-4e8e-8e00-cae769970934)

## Train and Test value:
![image](https://github.com/user-attachments/assets/f44a539e-7824-467f-a1bb-860eeb49cb1a)

## y_pred and y_test value:
![image](https://github.com/user-attachments/assets/41501479-1f65-4c6b-9224-2ba6d7bc127e)

## Training  set:
![image](https://github.com/user-attachments/assets/a972c7fd-ca5d-4fbd-b5c9-499d9a4bfa1e)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
