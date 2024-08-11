from sklearn.cluster import Birch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\10_7717_peerj_5665_dataYM2018_neuroblastoma.csv')

dataset = dataset.dropna()

print(dataset.info())

features_ = dataset.drop(columns=['time_months', 'outcome'])

encoder = LabelEncoder()
for col in features_.select_dtypes(include=['object']).columns:
    features_[col] = encoder.fit_transform(features_[col])

print(features_.info())

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_)

pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

birch = Birch(threshold=0.2, n_clusters=3) #Ho provato diversi valori per branching_factor ma non cambia nulla, mentre per threshold ho notato che impostandolo a 0.2 si ha un miglior clustering. Inoltre ho settato n_cluster=3 (default) poichè settato a None vengono generati 18 cluster
birch.fit(features)

labels = birch.labels_

features_['Cluster'] = labels

plot.figure(figsize=(12,8))
plot.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
plot.title("BIRCH Clustering")
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

heatMap = features_.groupby(['Cluster']).mean()

plot.figure(figsize=(30,10))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plot.xlabel("Cluster")
plot.ylabel("Features")
plot.title("HeatMap")
plot.show()