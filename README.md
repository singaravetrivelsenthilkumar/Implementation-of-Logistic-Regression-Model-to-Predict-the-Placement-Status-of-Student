# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. start program

Step 2. import the standard liabraries.

Step 3. upload the dataset and check for any null or duplicated values using.isnull() and .duplicated() function respectively

Step 4. import LabelEncoder and encode the dataset

Step 5. Import LogisticRegression from sklearn and aplly the model on the dataset

Step 6. Predict the values of array.

Step 7. Calculate the accuracy, confusion and classification report by imporving the required modules from sklearn.

Step 8. Apply new unknown values.

Step 9. stop program

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by:SINGARAVETRIVEL S

 RegisterNumber: 212222220048
```
import pandas as pd
data=pd.read_csv("C:/Users/SEC/Downloads/Placement_Data(1).csv")
data.head()
```

```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
```
data1.isnull()
```
```
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```
```
x=data1.iloc[:,1:-1]
x
```
```
y=data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train_y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
```
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## Output:

<H3>PLACEMENT DATA</H3>

![image](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/bab264cf-5188-4fbc-aac2-59687fb40e35)

<H3>SALARY DATA</H3>

![2](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/a910824f-dc86-40c2-9e3e-01b1880e7cde)

<H3>CHECKING THE NULL() FUNCTION</H3>

![Screenshot 2024-03-11 143207](https://github.com/maheshrajpurohit18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118749665/910fc18e-8775-499e-98d3-bd519783b7e1)


<H3>DATA DUPLICATE</H3>

![4](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/b516c2cd-f5d3-4b4e-b9e4-711bd86318ba)

<H3>PRINT DATA</H3>

![5](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/e6c3ca39-582d-4c1b-b42a-2fec1fe4458c)

<H3>DATA STATUS</H3>

![image](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/229ed0cd-e514-4269-8b99-4e24f749caed)

<H3>Y ARRAY</H3>

![7](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/9d08f30e-079b-4a69-931a-7dca39086650)

<H3>Y-PREDICTION ARRAY</H3>

![8](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/c101e699-8f25-44bd-b6a8-18e24b0058b6)
<H3>ACCURACY VALUE</H3>


![9](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/ddabf2ce-7202-4e66-a810-fe3599ffbea5)

<H3>CONFUSION VALUE</H3>


![10](https://github.com/premkumarkarthikeyan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476243/2393d6e1-4585-48ea-b8e0-55bf567cc78b)

<H3>CLASSIFICATION REPORT</H3>

![Screenshot 2024-03-11 144549](https://github.com/KotiSaiSankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344248/d00d2e49-ed49-46bd-b4c4-2a200bcfa21a)

<H3>PREDICTION OF LR</H3>

![Screenshot 2024-03-11 143756](https://github.com/KotiSaiSankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344248/945b020c-0bc8-4553-86c8-864d4513536f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
