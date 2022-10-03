'''
Chosen Data: Wine Quality dataset (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) was selected for
unsupervised learning model. Dataset has 13 columns and 1143 rows with no null values present.
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

mydata=pd.read_csv('./WineQT.csv')
mydata5=mydata[mydata['quality']==5]


mydata['fixed acidity'].describe()
sns.histplot(x = 'fixed acidity' , hue = 'quality',
             multiple = 'stack',data=mydata)
plt.title('Fixed Acidity')
plt.ylabel('Counts')
plt.xlabel('Fixed Acidity Values')
plt.show()

mydata['volatile acidity'].describe()
sns.histplot(x = 'volatile acidity' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Volatile Acidity')
plt.ylabel('Counts')
plt.xlabel('Volatile Acidity Values')
plt.show()

mydata['citric acid'].describe()
sns.histplot(x = 'citric acid' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Citric Acid')
plt.ylabel('Counts')
plt.xlabel('Citric Acid Values')
plt.show()

mydata['residual sugar'].describe()
sns.histplot(x = 'residual sugar' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Residual Sugar')
plt.ylabel('Counts')
plt.xlabel('Residual Sugar')
plt.show()

mydata['chlorides'].describe()
sns.histplot(x = 'chlorides' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Chlorides')
plt.ylabel('Counts')
plt.xlabel('Chloride')
plt.show()

mydata['free sulfur dioxide'].describe()
sns.histplot(x = 'free sulfur dioxide' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Free Sulfur Dioxide')
plt.ylabel('Counts')
plt.xlabel('Free Sulfur Dioxide')
plt.show()


mydata['total sulfur dioxide'].describe()
sns.histplot(x = 'total sulfur dioxide' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Total Sulfur Dioxide')
plt.ylabel('Counts')
plt.xlabel('Total Sulfur Dioxide')
plt.show()

mydata['density'].describe()
sns.histplot(x = 'density' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Density')
plt.ylabel('Counts')
plt.xlabel('Density')
plt.show()

mydata['pH'].describe()
sns.histplot(x = 'pH' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('pH')
plt.ylabel('Counts')
plt.xlabel('pH')
plt.show()

mydata['sulphates'].describe()
sns.histplot(x = 'sulphates' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Sulphates')
plt.ylabel('Counts')
plt.xlabel('Sulphate Values')
plt.show()

mydata['alcohol'].describe()
sns.histplot(x = 'alcohol' , hue = 'quality',
             multiple = 'stack',data=mydata5)
plt.title('Alcohol')
plt.ylabel('Counts')
plt.xlabel('Alcohol Values')
plt.show()

mydata['quality'].describe()
mydata['quality'].value_counts()
plt.hist(mydata['quality'], bins=5)
plt.title('Quality')
plt.ylabel('Counts')
plt.xlabel('Quality')
plt.show()

mydata['Id'].describe()
plt.hist(mydata['Id'], bins=15)
plt.title('Id')
plt.ylabel('Counts')
plt.xlabel('Id')
plt.show()

plt.hist(mydata['fixed acidity'], bins=10)
plt.title('Fixed Acidity')
plt.ylabel('Counts')
plt.xlabel('Fixed Acidity Values')
plt.show()


x=mydata[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
          'pH', 'sulphates', 'alcohol']]
y=mydata[['quality']]
y['quality'].replace(3, 0, inplace=True)
y['quality'].replace(4, 1, inplace=True)
y['quality'].replace(5, 2, inplace=True)
y['quality'].replace(6, 3, inplace=True)
y['quality'].replace(7, 4, inplace=True)
y['quality'].replace(8, 5, inplace=True)



scaler = StandardScaler()
wines_scaled = scaler.fit_transform(x)

hist_scaled = pd.DataFrame(wines_scaled, columns=x.columns).hist(figsize=(15, 10), bins=20)


#knn
numClusters = list(range(1, 11))
sse = []
wines_predictions = []
for k in numClusters:
    kmeans = KMeans(n_clusters=k, random_state=0)
    wines_predictions.append(kmeans.fit_predict(wines_scaled))
    sse.append(kmeans.inertia_)

# SSE/ Elbow
sns.lineplot(x=numClusters, y=sse)


kmeans = KMeans(n_clusters=6, random_state=0)
wp=kmeans.fit_predict(wines_scaled)
wp=pd.DataFrame(wp)
wp.value_counts()
wp1=pd.concat([y, wp], axis=1)
wp1.columns=['Quality', 'Predictions']

#HAC
from sklearn.cluster import AgglomerativeClustering
### BEGIN SOLUTION
ag = AgglomerativeClustering(n_clusters=6, linkage='ward', compute_full_tree=True)
wp2 = ag.fit_predict(x)
wp2=pd.DataFrame(wp2)
wp3=pd.concat([y, wp2], axis=1)
wp3.columns=['Quality', 'Predictions']
wp3['Predictions'].value_counts()

# Reduce the data with PCA into 6 components.
# It could be enough to use 2, but check 6 components for genuine interest.
pca6 = PCA(n_components=6)
wines_pca6 = pca6.fit_transform(wines_scaled)
wines_pca6=pd.DataFrame(wines_pca6)
hist_scaled = pd.DataFrame(wines_pca6).hist(figsize=(15, 10), bins=20)

#knn with pca
kmeans = KMeans(n_clusters=6, random_state=0)
wp4=kmeans.fit_predict(wines_pca6)
wp4=pd.DataFrame(wp4)
wp4.value_counts()
wp5=pd.concat([y, wp4], axis=1)
wp5.columns=['Quality', 'Predictions']
#hac
ag = AgglomerativeClustering(n_clusters=6, linkage='ward', compute_full_tree=True)
wp2 = ag.fit_predict(wines_pca6)
wp2=pd.DataFrame(wp2)
wp3=pd.concat([y, wp2], axis=1)
wp3.columns=['Quality', 'Predictions']
wp3['Predictions'].value_counts()




# How much of the variation of the data do the two principal components explain?
print('PCA 6 variance ratios: {}'.format(pca6.explained_variance_ratio_))

'''
Conclusions and Next Steps:

It was seen that dimensionality reduction combined with KNN clustering gave the best results when k was set to 6 which 
was the same number of clusters observed in the dataset. If the quality classification of the wine data wasn’t taken to 
account, various other k numbers can be tried to cluster the wine. For this elbow method could have been used to 
properly select the k value to see how many clusters can be obtained that were clearly separated. Further modeling can 
also be done with different dimensionality reduction parameters to prevent overfitting. 	

'''

