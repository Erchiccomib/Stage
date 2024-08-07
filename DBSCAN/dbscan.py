from sklearn.cluster import DBSCAN
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns


dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv')
dataset = dataset.dropna()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(dataset)

pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

dbscan = DBSCAN(eps=0.4, min_samples=5) 
#Ho provato, utilizzando la normalizzazione tramite MinMaxScaler, diversi epsilon ma non cambia il punteggio delle metriche
#Tramite la normalizzazione StandardScaler non si ottengono dei risultati più vantaggiosi, dunque è meglio tramite MinMaxScaler
dbscan.fit(features)

labels = dbscan.labels_

dataset['Cluster'] = labels

if len(set(labels)) > 1:
    silhouette_punteggio = silhouette_score(features, labels)
    print(f'Punteggio Silhouette: {silhouette_punteggio}')
    calinski_punteggio = calinski_harabasz_score(features, labels)
    print(f'Punteggio Calinski: {calinski_punteggio}')
    davies_punteggio = davies_bouldin_score(features, labels)
    print(f'Punteggio Davies: {davies_punteggio}')
else:
    print("Impossibile calcolare il punteggio delle metriche di valutazione con un solo cluster.")

# Visualizzazione dei risultati del clustering
plt.figure(figsize=(12, 8))
scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1 (Normalized)')
plt.ylabel('Feature 2 (Normalized)')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

##Visualizzazione HeatMap

heatMap = dataset.groupby('Cluster').mean()

plt.figure(figsize=(12,8))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plt.title('Heatmap of Feature Means by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()
