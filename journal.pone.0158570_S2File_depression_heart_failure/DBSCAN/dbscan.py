from sklearn.cluster import DBSCAN
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import GridSearchCV

#Leggo il dataset
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0158570_S2File_depression_heart_failure.csv')

features_ = dataset.drop(columns=['id'])

#Normalizzo i valori tramite MinMaxScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_)

#Riduco la dimensione dei valori del dataset per ottenere sia una migliore rappresentazione che un miglior clustering
pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

#Eseguo l'algoritmo DBSCAN
dbscan = DBSCAN(eps=0.9, min_samples=100) #Ho provato, utilizzando la normalizzazione tramite MinMaxScaler, diversi epsilon ma non cambia il punteggio delle metriche
dbscan.fit(features)

labels = dbscan.labels_

features_['Cluster'] = labels

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

heatMap = features_.groupby('Cluster').mean()

plt.figure(figsize=(12,8))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plt.title('Heatmap')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()
