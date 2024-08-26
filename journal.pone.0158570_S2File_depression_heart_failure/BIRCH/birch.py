from sklearn.cluster import Birch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

#Leggo il dataset
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0158570_S2File_depression_heart_failure.csv')

#Elimino le righe in cui mancano valori
dataset = dataset.dropna()


#Noto che tutte le colonne sono numeriche quindi non ho bisogno di conversioni
print(dataset.info())

features_ = dataset.drop(columns=['id'])

#Normalizzo i valori
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_)

#Setto PCA per ridurre la dimensionalità dei dati così da ottenere una migliora rappresentazione, setto n_components=2 per indicare le componenti da mantenere dopo la riduzione
pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

#Eseguo Birch
birch = Birch(threshold=1.4, n_clusters=None) #Ho provato diversi valori per branching_factor ma non cambia nulla, mentre per threshold ho notato che impostandolo a 1.4 si ha un miglior clustering
birch.fit(features)

labels = birch.labels_

features_['Cluster'] = labels

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
heatMap = features_.groupby(['Cluster']).mean()

plot.figure(figsize=(12,8))
sns.heatmap(heatMap.T, annot=True, cmap='viridis')
plot.xlabel("Cluster")
plot.ylabel("Features")
plot.title("HeatMap")
plot.subplots_adjust(left=0.3, right=0.8)
plot.show()