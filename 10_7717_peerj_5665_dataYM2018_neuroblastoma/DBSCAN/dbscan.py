from sklearn.cluster import DBSCAN
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns


dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\10_7717_peerj_5665_dataYM2018_neuroblastoma.csv')
dataset = dataset.dropna()

features_ = dataset.drop(columns=['time_months', 'outcome'])

encoder = LabelEncoder()
for col in features_.select_dtypes(include=['object']).columns:
    features_[col] = encoder.fit_transform(features_[col])


scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_)

pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

dbscan = DBSCAN(eps=0.5) 
dbscan.fit(features)

labels = dbscan.labels_

features_['Cluster'] = labels

if len(set(labels)) > 1:
    silhouette_punteggio = silhouette_score(features, labels)
    print(f'Punteggio Silhouette: {silhouette_punteggio}')
    calinski_punteggio = calinski_harabasz_score(features, labels)
    print(f'Punteggio Calinski: {calinski_punteggio}')
    davies_punteggio = davies_bouldin_score(features, labels)
    print(f'Punteggio Davies: {davies_punteggio}')
else:
    print("Errore sul numero dei cluster")

# Visualizzazione dei risultati del clustering
plt.figure(figsize=(12, 8))
scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('DBSCAN Clustering (MinMaxScaler)')
plt.xlabel('Component 1 PCA')
plt.ylabel('Component 2 PCA')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

#Visualizzazione HeatMap

heatMap = features_.groupby('Cluster').mean()

plt.figure(figsize=(12,8))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plt.title('Heatmap (MinMaxScaler)')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()
