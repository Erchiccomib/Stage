from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
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

mean = MeanShift(bandwidth=1.5, bin_seeding=True)
mean.fit(features)

labels = mean.labels_
features_['Cluster'] = labels

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