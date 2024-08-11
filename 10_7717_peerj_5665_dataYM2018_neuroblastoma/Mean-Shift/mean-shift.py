from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\10_7717_peerj_5665_dataYM2018_neuroblastoma.csv')

dataset = dataset.dropna()

features_ = dataset.drop(columns=['outcome','time_months'])

encoder = LabelEncoder()

for column in features_.select_dtypes(include=['object']).columns:
    features_[column] = encoder.fit_transform(features_[column])


scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_)

pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

mean = MeanShift(bandwidth=1)
mean.fit(features)

labels = mean.labels_
features_['Cluster'] = labels

#Punteggio Silhouette: 0.5699163842004293
#Punteggio Calinski: 203.8494758362625
#Punteggio Davies: 0.6880645237060501

plt.figure(figsize=(12,8))
plt.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
plt.xlabel("Component 1 PCA")
plt.ylabel("Component 2 PCA")
plt.grid(True)
plt.title("Mean-Shift Clustering")
plt.show()

punteggio_silhouette = silhouette_score(features, labels)
print(f'Punteggio Silhouette: {punteggio_silhouette}')

punteggio_calinski = calinski_harabasz_score(features, labels)
print(f'Punteggio Calinski: {punteggio_calinski}')

punteggio_davies = davies_bouldin_score(features, labels)
print(f'Punteggio Davies: {punteggio_davies}')

heatMap = features_.groupby(['Cluster']).mean()

plt.figure(figsize=(12,8))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plt.xlabel("Cluster")
plt.ylabel("Features")
plt.title("HeatMap")
plt.show()
