'''
Weather conditions in Szeged between the years 2006 and 2016 was selected for this project and the dataset
was obtained from Kaggle (https://www.kaggle.com/datasets/budincsevity/szeged-weather).
There are 96453 rows in the dataset with null values present in precipitation type.
Dataset was already cleaned where dirty and null data was removed.
Not every feature of the dataset will be used in this project and only Temperature and
Apparent Temperature features will be utilized in the prediction of the Humidity via regression models.
'''


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

mydata=pd.read_csv('./weatherHistory.csv')
#General Overview of the Data
print(mydata.head())
print(mydata.info())
#Checking the presence of null values
print(mydata.isnull().sum())

#Exploratory Data Analysis

#Replacing the missing values in precipitation type
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(mydata[['Precip Type']])
mydata[['Precip Type']] = imputer.transform(mydata[['Precip Type']])

# split column into multiple columns by delimiter

Data_Date=mydata['Formatted Date'].str.split(' ', expand=True)
Data_Date.columns = ['Date', 'Time', 'Time Zone']
Data_Date1=Data_Date['Date'].str.split('-', expand=True)
Data_Date1.columns = ['Year', 'Month', 'Day']

mydata = pd.concat([Data_Date1, mydata], axis=1, join='inner')
mydata2007=mydata[mydata['Year']=='2007']
mydata2007.info()

#Unsupervised Learning
#knn
temp=mydata2007[['Precip Type','Humidity','Temperature (C)','Apparent Temperature (C)']]
temp['Precip Type'].replace('rain', 1, inplace=True)
temp['Precip Type'].replace('snow', 2, inplace=True)

temp_rain=temp[temp['Precip Type']==1]
temp_snow=temp[temp['Precip Type']==2]
plt.scatter(range(0, len(temp_rain)), temp_rain['Temperature (C)'], label='Temperature', c='navy')
plt.scatter(range(0, len(temp)), temp_snow, label='Temperature', c='salmon')
plt.scatter(range(0, len(temp)), sorted(temp['Apparent Temperature (C)']), label= 'Apparent Temperature')
plt.scatter(range(0, len(temp)), temp['Humidity'], label= 'Humidity')
plt.legend()
plt.show()

#Weather Summary

sns.catplot(x='Summary', data=mydata ,kind="count")








mydata['Precip Type'].replace('rain', 0, inplace=True)
mydata['Precip Type'].replace('snow', 1, inplace=True)

mydata=mydata.drop(['Summary', 'Daily Summary'], axis = 1)
print(mydata.info())


temp=mydata[['Humidity','Temperature (C)','Apparent Temperature (C)']]
x=temp.drop('Humidity',axis=1)
y=temp['Humidity']

temp1=sorted(temp['Temperature (C)'])
plt.scatter(range(0, len(temp)), temp1, label='Temperature')
plt.scatter(range(0, len(temp)), sorted(temp['Apparent Temperature (C)']), label= 'Apparent Temperature')
plt.scatter(range(0, len(temp)), temp['Humidity'], label= 'Humidity')
plt.legend()
plt.show()

plt.plot(sorted(temp['Humidity']), label='Humidity')
plt.legend()
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#feature scaling
x_train=np.array(x_train)
x_test=np.array(x_test)


sc = MinMaxScaler()
x_test[:, :] = sc.fit_transform(x_test[:, :])
x_train[:, :] = sc.fit_transform(x_train[:, :])

#Linear regression to predict the humidity based on the real and felt temperature
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred_train=lr.predict(x_train)
y_pred_test=lr.predict(x_test)

from sklearn.metrics import r2_score
print('R2 score for train data: ', r2_score(y_train,y_pred_train))
print('R2 score for test data: ', r2_score(y_test,y_pred_test))

plt.plot(sorted(y_test), label='Real Humidity')
plt.plot(sorted(y_pred_test), label= 'Predicted Humidity')
plt.legend()
plt.show()

poly=PolynomialFeatures(degree=3)
poly_df=poly.fit_transform(x_test)
lr_poly=LinearRegression()
lr_poly.fit(poly_df,y_test)
y_pred_poly=lr_poly.predict(poly_df)

plt.plot(sorted(y_test), label='Real Humidity')
plt.plot(sorted(y_pred_poly), label= 'Predicted Humidity')
plt.legend()
plt.show()




