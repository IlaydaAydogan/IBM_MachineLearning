'''
Chosen Data: Wine Quality dataset (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) was selected for
deep learning models. Dataset has 13 columns and 1143 rows with no null values present. Only 1 quality group was
selected which is the quality group of 5 for analysis. When this filter was applied only 483 of the wine were left
for analysis.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models  import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.decomposition import PCA
from pyod.models.auto_encoder import AutoEncoder

mydata=pd.read_csv('./WineQT.csv')
mydata5=mydata[mydata['quality']==5]
index=mydata5['Id']

mydata5=mydata5.drop(['fixed acidity','volatile acidity', 'citric acid', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'quality', 'Id'], axis = 1)

#feature scaling
sc = MinMaxScaler()
mydata5scale=np.array(mydata5)
mydata5scale=mydata5scale.reshape(-1,4)
mydata5scale=sc.fit_transform(mydata5scale)
mydata5_final=pd.DataFrame(mydata5scale, columns=['residual sugar', 'chlorides', 'sulphates', 'alcohol'])


sns.pairplot(mydata5_final, size=2.5);

#PCA
pca1 = PCA(n_components=1)
wines_pca1 = pca1.fit_transform(mydata5_final)
wines_pca1=pd.DataFrame(wines_pca1)

#test train split
X_train,X_test =train_test_split(wines_pca1,test_size=0.2,random_state=123)
X_train= X_train.sort_index()
X_test=X_test.sort_index()

test_indices = X_test.index.tolist(); train_indices = X_train.index.tolist()

x_train=np.array(X_train)
x_test=np.array(X_test)


#Model 1

model_1 = Sequential()
model_1.add(Dense(3,input_shape = (1,),activation = 'sigmoid'))
model_1.add(Dense(1,activation='sigmoid'))

model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mean_squared_error'])


history = model_1.fit(x_train, x_train, shuffle=True, epochs=15, validation_data=(x_test, x_test))

train_res=model_1.predict(x_train); test_res=model_1.predict(x_test)
train_res=list(train_res[0:,0].flatten()); test_res=list(test_res[0:,0].flatten())

volt_train=X_train.values.tolist(); volt_test=X_test.values.tolist()

train_results = pd.DataFrame({"Index": train_indices,"OG Data": volt_train ,"Predictions": train_res})
test_results=pd.DataFrame({"Index": test_indices,"OG Data": volt_test ,"Predictions": test_res})
print(train_results.describe()); print(test_results.describe())

final_data= pd.concat([train_results, test_results])
final_data = final_data.sort_values(by=['Index'])
final_data= final_data.reset_index(drop=True)
print(final_data.describe())

p1_threshold=final_data[final_data['Predictions']<final_data['Predictions'].quantile(q=0.05)]
threshold_value_down=p1_threshold['Predictions'].mean()-final_data['Predictions'].std()
ad_thresh_down=final_data[final_data['Predictions']<threshold_value_down]
ad_thresh1=ad_thresh_down.index.tolist()

p99_threshold=final_data[final_data['Predictions']>final_data['Predictions'].quantile(q=0.95)]
threshold_value_up=p99_threshold['Predictions'].mean()+final_data['Predictions'].std()
ad_thresh_up=final_data[final_data['Predictions']>threshold_value_up]
ad_thresh2=ad_thresh_up.index.tolist()

ad_thresh=ad_thresh1+ad_thresh2
anomaly_info=final_data.iloc[ad_thresh]

sc_n=plt.scatter(x='Index', y='Predictions', color='salmon', data= final_data )
sc1 = plt.scatter('Index', 'Predictions', color='navy', data=anomaly_info)
plt.legend(['Expected','Anomaly'])
plt.xlabel('Index')
plt.ylabel('Predictions')
plt.title('Anomalies Observed in the Wine Quality Group 5')
plt.show()


#model 2

#test train split
X_train,X_test =train_test_split(mydata5_final,test_size=0.2,random_state=123)
X_train= X_train.sort_index()
X_test=X_test.sort_index()

test_indices = X_test.index.tolist(); train_indices = X_train.index.tolist()

x_train=np.array(X_train); x_train=x_train[:, 0]
x_test=np.array(X_test); x_test=x_test[:, 0]


model_1 = Sequential()
model_1.add(Dense(4,input_shape = (1,),activation = 'sigmoid'))
model_1.add(Dense(1,activation='sigmoid'))

model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mean_squared_error'])


history = model_1.fit(x_train, x_train, shuffle=True, epochs=15, validation_data=(x_test, x_test))

train_res=model_1.predict(x_train); test_res=model_1.predict(x_test)
train_res=list(train_res[0:,0].flatten()); test_res=list(test_res[0:,0].flatten())

volt_train=X_train.values.tolist(); volt_test=X_test.values.tolist()

train_results = pd.DataFrame({"Index": train_indices,"OG Data": volt_train ,"Predictions": train_res})
test_results=pd.DataFrame({"Index": test_indices,"OG Data": volt_test ,"Predictions": test_res})
print(train_results.describe()); print(test_results.describe())

final_data= pd.concat([train_results, test_results])
final_data = final_data.sort_values(by=['Index'])
final_data= final_data.reset_index(drop=True)
print(final_data.describe())

p1_threshold=final_data[final_data['Predictions']<final_data['Predictions'].quantile(q=0.05)]
threshold_value_down=p1_threshold['Predictions'].mean()-final_data['Predictions'].std()
ad_thresh_down=final_data[final_data['Predictions']<threshold_value_down]
ad_thresh1=ad_thresh_down.index.tolist()

p99_threshold=final_data[final_data['Predictions']>final_data['Predictions'].quantile(q=0.95)]
threshold_value_up=p99_threshold['Predictions'].mean()+final_data['Predictions'].std()
ad_thresh_up=final_data[final_data['Predictions']>threshold_value_up]
ad_thresh2=ad_thresh_up.index.tolist()

ad_thresh=ad_thresh1+ad_thresh2
anomaly_info=final_data.iloc[ad_thresh]

sc_n=plt.scatter(x='Index', y='Predictions', color='salmon', data= final_data )
sc1 = plt.scatter('Index', 'Predictions', color='navy', data=anomaly_info)
plt.legend(['Expected','Anomaly'])
plt.xlabel('Index')
plt.ylabel('Predictions')
plt.title('Anomalies Observed in the Wine Quality Group 5')
plt.show()


#model 3: Autoencoder
#modelling
clf1 = AutoEncoder(hidden_neurons =[1,1], epochs=5, contamination=0.05)
ad1=clf1.fit(x_train)

#threshold value set for anomaly score threshold by the model
thresh= clf1.threshold_

#getting prediction scores and anomaly values
y_train_scores= clf1.decision_function(x_train); y_train_pred = clf1.predict(x_train)
y_test_scores = clf1.decision_function(x_test); y_test_pred = clf1.predict(x_test)

train_results = pd.DataFrame({"Index": train_indices,"Scores": y_train_scores ,"Predictions": y_train_pred})
train_results = train_results.sort_values(by=['Index'])
test_results = pd.DataFrame({"Index": test_indices ,"Scores": y_test_scores ,"Predictions": y_test_pred})
test_results = test_results.sort_values(by=["Index"])


final_data=pd.concat([train_results,test_results])
final_data=final_data.sort_values(by=['Index'])
final_data=final_data.reset_index(drop=True)

ano1=final_data[final_data['Predictions']==0]
ano2=final_data[final_data['Predictions']==1]

plt.scatter(ano1['Index'], ano1['Scores'], color='salmon')
plt.scatter(ano2['Index'], ano2['Scores'], color='navy')
plt.legend(['Expected','Anomaly'])
plt.xlabel('Index')
plt.ylabel('Predictions')
plt.title('Anomalies Observed in the Wine Quality Group 5')
plt.show()


'''
Conclusions and Next Steps:
Various deep learning methods were used to detect the outliers that were present within the wine quality group of 5. 
Initial model was a Keras model without dimension reduction and only 1 anomaly was detected. Second model was the Keras 
with dimension reduction and detected 7 outliers. Dimension reduction was done via PCA to represent all 4 variables in 
1. Final model was an autoencoder and it detected 21 anomalies. From the graphs created following outlier detection, it was 
seen that autoencoder model was better at predicting the outliers. Future moves can include the utilization of deeper 
neural networks as well as more sophisticated recurrent networks to better capture the anomalies.
'''
