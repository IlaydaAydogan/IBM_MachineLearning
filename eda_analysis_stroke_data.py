'''Brain stroke data was obtained from Kaggle
(https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset?select=brain_stroke.csv).
There are various risk factors that contribute to the brain stroke onset and decreasing the risk
factors tend to lower the frequency of brain strokes in individuals. Risk factors such as history
of smoking or age can be used to predict the likeliness for a stroke to occur. The dataset contains
11 columns with the last column is in regards to whether the individual had a stroke or not. The first
10 columns are features that are believed to be related to onset of strokes. There are 4981 samples in
the dataset with no null values present. Dataset was already cleaned where dirty and null data was removed.
'''

import pandas as pd
import numpy as np
from scipy import stats
import math
import seaborn as sns
import matplotlib.pyplot as plt

brain_data=pd.read_csv('./brain_stroke.csv')
print('Number of rows in the brain stroke data is ', len(brain_data))
print('Number of columns in the stroke data is ', len(brain_data.columns))

#Data types for each column and how many of each column is filled can be observed via
print(brain_data.info())
#Which columns in the dataframe that weren't filled/ was empty
print('Which columns in the dataframe that werent filled/ was empty: ', brain_data.isnull().sum())
print('Names of the columns in a list format: ', brain_data.columns.tolist())

####
#Gender Distribution in the Dataset
gend=[len(brain_data[brain_data['gender']=='Male']),len(brain_data[brain_data['gender']=='Female'])]
plt.bar(brain_data['gender'].unique(), gend)
plt.title('Gender Distribution of Samples')
plt.xlabel('Gender')
plt.ylabel('# of Samples')
plt.show()
print("{:.2f}".format((len(brain_data[brain_data['gender']=='Male'])/len(brain_data))*100), '% of the stroke data sample is male and',
      "{:.2f}".format((len(brain_data[brain_data['gender']=='Female'])/len(brain_data))*100),'% of the stroke data sample is female.')

#Gender in relation to Stroke
sns.catplot(x='stroke',hue='gender',data=brain_data ,kind="count")
plt.xlabel('Stroke')
plt.ylabel('# of Samples')
plt.show()
####

print('Minimum age present to a sample in the dataset is ', "{:.1f}".format(min(brain_data['age'])))
print('Maximum age present to a sample in the dataset is ', "{:.1f}".format(max(brain_data['age'])))
print('Average age of samples is ', "{:.2f}".format(sum(brain_data['age'])/len(brain_data)))

plt.hist([brain_data[brain_data['gender']=='Male']['age'], brain_data[brain_data['gender']=='Female']['age']],
         10, label=['Males', 'Females'])
plt.legend(loc='upper right')
plt.title('Age Distribution of Samples')
plt.xlabel('Age')
plt.ylabel('# of Samples')
plt.show()

sns.histplot(data=brain_data, x="age", hue="stroke", multiple="stack")
plt.title('Age Distribution of Samples')
plt.xlabel('Age')
plt.ylabel('# of Samples')
plt.show()



###

print(len(brain_data[brain_data['hypertension']==1]), ' of the patients have hypertension.')
print("{:.2f}".format((len(brain_data[brain_data['hypertension']==1])/len(brain_data))*100), '% of the patients have hypertension.')

sns.catplot(x='hypertension',hue='gender',data=brain_data ,kind="count").set(title='Presence of Hypertension in Patients')
plt.xlabel('Hypertension Presence')
plt.ylabel('# of Samples')
plt.show()

sns.catplot(x='hypertension',hue='stroke',data=brain_data ,kind="count").set(title='Presence of Hypertension in Patients')
plt.xlabel('Hypertension Presence')
plt.ylabel('# of Samples')
plt.show()
#####
print(len(brain_data[brain_data['heart_disease']==1]), ' of the patients have heart disease.')
print("{:.2f}".format((len(brain_data[brain_data['heart_disease']==1])/len(brain_data))*100), '% of the patients have heart disease.')

sns.catplot(x='heart_disease',hue='gender',data=brain_data ,kind="count").set(title='Presence of Heart Disease in Patients')
plt.xlabel('Heart Disease Presence')
plt.ylabel('# of Samples')
plt.show()

sns.catplot(x='heart_disease',hue='stroke',data=brain_data ,kind="count").set(title='Presence of Hypertension in Patients')
plt.xlabel('Heart Disease Presence')
plt.ylabel('# of Samples')
plt.show()
####
print(len(brain_data[brain_data['ever_married']=='Yes']), ' of the patients have married.')
print("{:.2f}".format((len(brain_data[brain_data['ever_married']=='Yes'])/len(brain_data))*100), '% of the patients have married.')

sns.catplot(x='ever_married',hue='gender',data=brain_data ,kind="count")
plt.xlabel('Marital Status')
plt.ylabel('# of Samples')
plt.show()

sns.catplot(x='ever_married',hue='stroke',data=brain_data ,kind="count")
plt.xlabel('Marital Status')
plt.ylabel('# of Samples')
plt.show()


stroke=brain_data[brain_data['stroke']==1]
print(len(stroke[stroke['ever_married']=='Yes']), ' of the patients have married.')

sns.catplot(x='ever_married',data=stroke ,kind="count").set(title='Marital Status of the Patients who Had a Stroke')
plt.xlabel('Marital Status')
plt.ylabel('# of Samples')
plt.show()

####
sns.catplot(x='work_type',data=brain_data ,kind="count").set(title='Work Type of the Patients in the Dataset')
plt.xlabel('Work Type')
plt.ylabel('# of Samples')
plt.show()


sns.histplot(data=brain_data, x="work_type", hue="stroke", multiple="stack")
plt.title('Work Type of Patients in Relation to Stroke')
plt.xlabel('Work Type')
plt.ylabel('# of Samples')
plt.show()

#####
sns.catplot(x='Residence_type',data=brain_data ,kind="count").set(title='Residence Type of the Patients in the Dataset')
plt.xlabel('Residence Type')
plt.ylabel('# of Samples')
plt.show()

sns.histplot(data=brain_data, x="Residence_type", hue="stroke", multiple="stack")
plt.title('Residence Type of Patients in Relation to Stroke')
plt.xlabel('Residence Type')
plt.ylabel('# of Samples')
plt.show()

####

sns.histplot(data=brain_data, x="avg_glucose_level", hue="stroke", multiple="stack")
plt.title('Blood Glucose Levels in Relation to Stroke')
plt.xlabel('Average Glucose Levels')
plt.ylabel('# of Samples')
plt.show()

sns.histplot(data=brain_data, x="bmi", hue="stroke", multiple="stack")
plt.title('BMI Levels in Relation to Stroke')
plt.xlabel('BMI')
plt.ylabel('# of Samples')
plt.show()

####

sns.histplot(data=brain_data, x="smoking_status", hue="stroke", multiple="stack")
plt.title('Smoking Status in Relation to Stroke')
plt.xlabel('Smoking Status')
plt.ylabel('# of Samples')
plt.show()

#####

#Hypothesis Testing
#null h0
#There is no significant correlation between brain stroke and average glucose levels
#h1
#There is a significant correlation between brain stroke and average glucose levels
stroke=brain_data[brain_data['stroke']==1]

from scipy.stats import pearsonr
data1=np.array(brain_data['avg_glucose_level'])
data2=np.array(brain_data['stroke'])
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')