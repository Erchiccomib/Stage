from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

##Leggo il dataset dal file csv
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\Takashi2019_diabetes_type1_dataset_preprocessed.csv')
dataset = dataset.dropna()

print(dataset.info()) ## Da qui noto tutte le informazioni dei dati, sopratutto il tipo di dato presente così da capire se pre-elaborare i dati

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(dataset)

pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

Z = linkage(features, 'ward') #Ho provato con 'average' ma si ottine un dendrogramma difficile da interpretare rispetto a quello generato con 'ward'

plot.figure(figsize=(10,7))
dendrogram(Z)
plot.title('Dendrogramma')
plot.xlabel('Cluster')
plot.ylabel('Distance')
plot.show()

agglomerative = AgglomerativeClustering(n_clusters=3, linkage="ward") #Siccome utilizzo il collegamento "ward" non posso settare l'affinity diversa da euclidean
agglomerative.fit(features)

labels= agglomerative.labels_

dataset['Cluster'] = labels

plot.figure(figsize=(12,8))
plot.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
plot.title("Agglomerative Clustering (k=3)")
plot.xlabel("Component 1 PCA")
plot.ylabel("Component 2 PCA")
plot.grid(True)
plot.show()

punteggio_silhouette = silhouette_score(features, labels)
print(f'Punteggio Silhouette: {punteggio_silhouette}')

punteggio_calinski = calinski_harabasz_score(features, labels)
print(f'Punteggio Calinski: {punteggio_calinski}')

punteggio_davies = davies_bouldin_score(features, labels)
print(f'Punteggio Davies: {punteggio_davies}')

#Stampo a video l'heatMap 
features_only = dataset.drop(columns=['Cluster'], errors='ignore') #Escludo la colonna Cluster

# Converto 'features_scaled' in un DataFrame Pandas con i nomi delle colonne originali
features_df = pd.DataFrame(features_scaled, columns=features_only.columns)

features_df['Cluster'] = dataset['Cluster'].values

cluster_summary = features_df.groupby('Cluster').mean()

plot.figure(figsize=(12,8))
sns.heatmap(cluster_summary.T, annot=True, cmap='viridis')
plot.xlabel("Cluster")
plot.ylabel("Features")
plot.title("HeatMap")
plot.show()