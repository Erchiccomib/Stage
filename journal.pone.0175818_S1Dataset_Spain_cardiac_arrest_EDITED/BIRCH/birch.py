from sklearn.cluster import Birch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

#Leggo il dataset
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0175818_S1Dataset_Spain_cardiac_arrest_EDITED..csv')

#Elimino le righe in cui mancano valori
dataset = dataset.dropna()

#Noto che tutte le colonne sono numeriche quindi non ho bisogno di conversioni
print(dataset.info())

#Normalizzo i valori
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(dataset)

#Setto PCA per ridurre la dimensionalità dei dati così da ottenere una migliora rappresentazione, setto n_components=2 per indicare le componenti da mantenere dopo la riduzione
pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

#Eseguo Birch
birch = Birch(threshold=0.5, n_clusters=None) #Ho provato diversi valori per branching_factor ma non cambia nulla, mentre per threshold ho notato che impostandolo a 0.5 si ha un miglior clustering
birch.fit(features)

labels = birch.labels_

dataset['Cluster'] = labels

#Mostro i cluster ottenuti
plot.figure(figsize=(12,8))
plot.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
plot.title("BIRCH Clustering")
plot.xlabel("Component 1 PCA")
plot.ylabel("Component 2 PCA")
plot.grid(True)
plot.show()

#Calcolo le metriche di valutazione e le stampo
punteggio_silhouette = silhouette_score(features, labels)
print(f'Punteggio Silhouette: {punteggio_silhouette}')

punteggio_calinski = calinski_harabasz_score(features, labels)
print(f'Punteggio Calinski: {punteggio_calinski}')

punteggio_davies = davies_bouldin_score(features, labels)
print(f'Punteggio Davies: {punteggio_davies}')

#Mostro l'heatMap
heatMap = dataset.groupby(['Cluster']).mean()

plot.figure(figsize=(30,10))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plot.xlabel("Cluster")
plot.ylabel("Features")
plot.title("HeatMap")
plot.show()