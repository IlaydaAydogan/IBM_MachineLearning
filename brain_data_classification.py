'''
Brain stroke data was obtained from Kaggle
(https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset?select=brain_stroke.csv ).
There are various risk factors that contribute to the brain stroke onset and decreasing the risk
factors tend to lower the frequency of brain strokes in individuals. Risk factors such as history
of smoking or age can be used to predict the likeliness for a stroke to occur.
The dataset contains 11 columns with the last column is in regards to whether the individual had
a stroke or not. The first 10 column are features that are believed to be related to onset of strokes.
There are 4981 samples in the dataset with no null values present. Dataset was already cleaned where dirty
and null data was removed.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

brain_data=pd.read_csv('./brain_stroke.csv')
#Dummy data for categorical values
#for sex
brain_data['gender']=brain_data['gender'].replace('Male', 1)
brain_data['gender']=brain_data['gender'].replace('Female', 2)
#marriage
brain_data['ever_married']=brain_data['ever_married'].replace('Yes', 1)
brain_data['ever_married']=brain_data['ever_married'].replace('No', 0)

#work type
brain_data['work_type']=brain_data['work_type'].replace('children', 1)
brain_data['work_type']=brain_data['work_type'].replace('Govt_job', 2)
brain_data['work_type']=brain_data['work_type'].replace('Self-employed', 3)
brain_data['work_type']=brain_data['work_type'].replace('Private', 4)

#residence type
brain_data['Residence_type']=brain_data['Residence_type'].replace('Urban', 1)
brain_data['Residence_type']=brain_data['Residence_type'].replace('Rural', 0)

#smoking status
brain_data['smoking_status']=brain_data['smoking_status'].replace('Unknown', 1)
brain_data['smoking_status']=brain_data['smoking_status'].replace('never smoked', 2)
brain_data['smoking_status']=brain_data['smoking_status'].replace('formerly smoked', 3)
brain_data['smoking_status']=brain_data['smoking_status'].replace('smokes', 4)
#Feature Scaling
mydata_x= brain_data.iloc[:, :-1].values
mydata_y= brain_data.iloc[:, -1].values

sc = StandardScaler()
mydata_x[:, [1,5,7,8,9]] = sc.fit_transform(mydata_x[:, [1,5,7,8,9]])

#Train/Test Split
# Get the split indexes
from sklearn.model_selection import train_test_split

#Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(mydata_x, mydata_y, test_size = 0.3, stratify=mydata_y)

#Classification with Logistic Regression

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lrmodel=lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)

y_test=pd.DataFrame(y_test)
y_pred=pd.DataFrame(y_pred)
y_acc=pd.concat([y_test, y_pred], axis=1)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Classification with Decision Trees
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred_dt = clf.predict(X_test)
y_pred_dt=pd.DataFrame(y_pred_dt)
y_acc_dt=pd.concat([y_test, y_pred_dt], axis=1)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_dt))
#Classification with Support Vector Machine (SVM)
from sklearn.ensemble import RandomForestClassifier

clf1 = RandomForestClassifier(max_depth=2, n_estimators=5, random_state=0)
clf1=clf1.fit(X_train, y_train)
y_pred_svm= clf1.predict(X_test)
y_pred_svm=pd.DataFrame(y_pred_svm)
y_acc_svm=pd.concat([y_test, y_pred_svm], axis=1)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svm))
y_pred_svm.value_counts()
'''
Conclusion: Although accuracy was higher compared to logistic classification, classification with decision tree 
is more preferable as we see positive values for stroke prediction which is absent in logistic classification. 
Next Steps:  As the data is unbalanced, further methodological approaches can be performed to account for the 
unbalance in the dataset. More sophisticated ensemble models can be used to increase accuracy and fine tune the model. 
Additionally, upsampling of the minority data or downsampling of the more dominant negative stroke data can be performed 
to get better predictions in the test group. A possibility is the usage of hybrid models to more effectively capture the 
features in the minority data. Also, only the most basic models were used for classification and more additions can be 
done to understand which features would have more weight in predicting brain strokes.
'''