from sklearn.cluster import MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\Takashi2019_diabetes_type1_dataset_preprocessed.csv')

dataset = dataset.dropna()

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(dataset)

pca = PCA()
features = pca.fit_transform(features_scaled)

mean = MeanShift(bandwidth=1.35)
mean.fit(features)

labels = mean.labels_
dataset['Cluster'] = labels

plt.figure(figsize=(12,8))
plt.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
plt.xlabel("Component 1 PCA")
plt.ylabel("Component 2 PCA")
plt.grid(True)
plt.title("Mean-Shift")
plt.show()

punteggio_silhouette = silhouette_score(features, labels)
print(f'Punteggio Silhouette: {punteggio_silhouette}')

punteggio_calinski = calinski_harabasz_score(features, labels)
print(f'Punteggio Calinski: {punteggio_calinski}')

punteggio_davies = davies_bouldin_score(features, labels)
print(f'Punteggio Davies: {punteggio_davies}')

heatMap = dataset.groupby(['Cluster']).mean()

plt.figure(figsize=(12,8))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plt.xlabel("Cluster")
plt.ylabel("Features")
plt.title("HeatMap")
plt.show()


