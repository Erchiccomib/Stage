from sklearn.cluster import DBSCAN
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns

#Leggo il dataset
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\Takashi2019_diabetes_type1_dataset_preprocessed.csv')

dataset = dataset.dropna()
#Normalizzo i valori tramite MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(dataset)

#Riduco la dimensione dei valori del dataset per ottenere sia una migliore rappresentazione che un miglior clustering
pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

#Eseguo l'algoritmo DBSCAN
dbscan = DBSCAN(eps=0.3) #Ho provato diverse configurazioni ottenendo un miglior riscontro con eps=0.3
dbscan.fit(features)

labels = dbscan.labels_

dataset['Cluster'] = labels

#Stampo i valori delle metriche di valutazione
silhouette_punteggio = silhouette_score(features, labels)
print(f'Punteggio Silhouette: {silhouette_punteggio}')

calinski_punteggio = calinski_harabasz_score(features, labels)
print(f'Punteggio Calinski: {calinski_punteggio}')
    
davies_punteggio = davies_bouldin_score(features, labels)
print(f'Punteggio Davies: {davies_punteggio}')

# Visualizzazione dei risultati del clustering
plt.figure(figsize=(12, 8))
scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('DBSCAN Clustering')
plt.xlabel('Component 1 PCA')
plt.ylabel('Component 2 PCA')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

#Visualizzazione HeatMap

heatMap = dataset.groupby('Cluster').mean()

plt.figure(figsize=(12,8))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plt.title('Heatmap')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()
