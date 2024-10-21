from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

##Leggo il dataset dal file csv
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\10_7717_peerj_5665_dataYM2018_neuroblastoma.csv')
dataset = dataset.dropna()

print(dataset.info()) ## Da qui noto tutte le informazioni dei dati, sopratutto il tipo di dato presente così da capire se pre-elaborare i dati

#Ho notato che ci sono sia colonne numeriche che colonne categoriali, dunque è meglio trasformare le colonne categoriali in numeriche

features_ = dataset.drop(columns=['outcome', 'time_months']) #Escludo outcome perchè pongo il clustering sui valori di base dunque su nessun valore 'finale', mentre per time_months ho deciso di escluderlo perchè non è facilmente interpretabile e ha una distribuzione troppo elevata

#Inizializzo l'encoder per trasformare i valori categoriali in valori numerici
encoder = LabelEncoder()
#Eseguo la trasformazione selezionando tutte le colonne categoriali per poi trasformarle mediante LabelEncoder 
for col in features_.select_dtypes(include=['object']).columns:
    features_[col] = encoder.fit_transform(features_[col])

#Normalizzo i valori
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_)

#Riduco la dimensione dei valori così da ottenere una migliore rappresentazione e un miglior clustering
pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

#Riduco la dimensione dei valori del dataset per ottenere sia una migliore rappresentazione che un miglior clustering
pca = PCA(n_components=2)
features = pca.fit_transform(features_scaled)

#specifico il criterio utilizzato per unire i cluster durante il processo di clustering gerarchico e salvo il collegamento in Z
Z = linkage(features, 'average')

#Mostro a video il dendrogramma per comprendere il numero di cluster da formare (k=4)
plot.figure(figsize=(10,7))
dendrogram(Z)
plot.title('Dendrogramma')
plot.xlabel('Cluster')
plot.ylabel('Distance')
plot.show()

#Eseguo il cluster con k=4
agglomerative = AgglomerativeClustering(n_clusters=4, linkage="average", metric="manhattan")
agglomerative.fit(features)

labels= agglomerative.labels_

features_['Cluster'] = labels

#Mostro il clustering ottenuto
plot.figure(figsize=(12,8))
plot.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
plot.title("Agglomerative Clustering (k=4)")
plot.xlabel("Component 1 PCA")
plot.ylabel("Component 2 PCA")
plot.grid(True)
plot.show()

#Calcolo e stampo i valori delle metriche di valutazione
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

plot.figure(figsize=(12,8))
sns.heatmap(cluster_summary.T, annot=True, cmap='viridis')
plot.xlabel("Cluster")
plot.ylabel("Features")
plot.title("HeatMap")
plot.show()