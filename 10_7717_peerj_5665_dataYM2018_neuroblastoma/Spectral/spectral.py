from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\10_7717_peerj_5665_dataYM2018_neuroblastoma.csv')

dataset = dataset.dropna()

encoder = LabelEncoder()

features_ = dataset.drop(columns=['outcome','time_months'])

for column in features_.select_dtypes(include=['object']).columns:
    features_[column] = encoder.fit_transform(features_[column])

scaler = MinMaxScaler()
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
plt.title('Gomito')
plt.grid(True)
plt.show()

spectral = SpectralClustering(n_clusters=3, affinity='rbf', gamma=10, random_state=42)
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

#Stampo a video l'heatMap 
features_only = features_.drop(columns=['Cluster'], errors='ignore') #Escludo la colonna Cluster

# Converto 'features_scaled' in un DataFrame Pandas con i nomi delle colonne originali
features_df = pd.DataFrame(features_scaled, columns=features_only.columns)

features_df['Cluster'] = features_['Cluster'].values

cluster_summary = features_df.groupby('Cluster').mean()

plt.figure(figsize=(12,8))
sns.heatmap(cluster_summary.T, annot=True, cmap='viridis')
plt.xlabel("Cluster")
plt.ylabel("Features")
plt.title("HeatMap")
plt.show()