from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

##Leggo il dataset dal file csv
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0158570_S2File_depression_heart_failure.csv')
dataset = dataset.dropna()

print(dataset.info()) ## Da qui noto tutte le informazioni dei dati, sopratutto il tipo di dato presente così da capire se pre-elaborare i dati

features_ = dataset.drop(columns="id")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_)

pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

inertia = []
K = range(2,10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

plot.figure(figsize=(10,6))
plot.plot(K, inertia, 'bx-')
plot.xlabel('Numeri di cluster (k)')
plot.ylabel('Inertia')
plot.title('Gomito')
plot.grid(True)
plot.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)

labels_kmeans = kmeans.labels_

centroids = kmeans.cluster_centers_

Z = linkage(centroids, 'ward') #Ho provato con 'average' ma si ottine un dendrogramma difficile da interpretare rispetto a quello generato con 'ward'

plot.figure(figsize=(10,7))
dendrogram(Z)
plot.title('Dendrogramma')
plot.xlabel('Cluster')
plot.ylabel('Distance')
plot.show()

agglomerative = AgglomerativeClustering(n_clusters=3, linkage="average") #Siccome utilizzo il collegamento "ward" non posso settare l'affinity diversa da euclidean
agglomerative.fit(centroids)

labels_agglomerative_on_centroids= agglomerative.labels_

labels_agglomerative = [labels_agglomerative_on_centroids[label] for label in labels_kmeans]

dataset['Cluster'] = labels_agglomerative

plot.figure(figsize=(12,8))
plot.scatter(features[:,0], features[:,1], c=labels_agglomerative, cmap='viridis')
plot.title("Agglomerative Clustering (k=4)")
plot.xlabel("Component 1 PCA")
plot.ylabel("Component 2 PCA")
plot.grid(True)
plot.show()

punteggio_silhouette = silhouette_score(features, labels_agglomerative)
print(f'Punteggio Silhouette: {punteggio_silhouette}')

punteggio_calinski = calinski_harabasz_score(features, labels_agglomerative)
print(f'Punteggio Calinski: {punteggio_calinski}')

punteggio_davies = davies_bouldin_score(features, labels_agglomerative)
print(f'Punteggio Davies: {punteggio_davies}')

heatMap = dataset.groupby(['Cluster']).mean()

plot.figure(figsize=(12,8))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plot.xlabel("Cluster")
plot.ylabel("Features")
plot.title("HeatMap")
plot.subplots_adjust(left=0.3, right=0.8)
plot.show()