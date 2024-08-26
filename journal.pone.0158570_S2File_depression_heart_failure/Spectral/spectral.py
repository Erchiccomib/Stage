from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0158570_S2File_depression_heart_failure.csv')

dataset = dataset.dropna()

features_ = dataset.drop(columns=['id'])

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_)

pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

inertia=[]
K=range(2,10)

for k in K:
    spectral = SpectralClustering(n_clusters=k, affinity='rbf', random_state=42)
    spectral.fit(features)
    labels = spectral.labels_
    centroids = np.array([features[labels == i].mean(axis=0) for i in range(k)])
    
    # Calcolo della "inerzia" come somma delle distanze quadrate dai centroidi
    closest, distances = pairwise_distances_argmin_min(features, centroids)
    inertia_ = np.sum(distances ** 2)
    inertia.append(inertia_)

# Visualizzazione del metodo del gomito
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, marker='o')
plt.xlabel('Numero di Cluster')
plt.ylabel('Inerzia')
plt.title('Metodo del Gomito')
plt.grid(True)
plt.show()

spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42, n_neighbors=40)
spectral.fit(features)

labels = spectral.labels_
features_['Cluster'] = labels

plt.figure(figsize=(12,8))
plt.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
plt.xlabel("Component 1 PCA")
plt.ylabel("Component 2 PCA")
plt.grid(True)
plt.title("Spectral Clustering (k=3)")
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
plt.subplots_adjust(left=0.3, right=0.8)
plt.show()
