import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
import seaborn as sns

##Leggo il dataset dal file csv
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\10_7717_peerj_5665_dataYM2018_neuroblastoma.csv')
dataset = dataset.dropna()

print(dataset.info()) ## Da qui noto tutte le informazioni dei dati, sopratutto il tipo di dato presente così da capire se pre-elaborare i dati

#Ho notato che ci sono sia colonne numeriche che colonne categoriali, dunque è meglio trasformare le colonne categoriali in numeriche

features_ = dataset.drop(columns=['time_months', 'outcome'])#Escludo outcome perchè pongo il clustering sui valori di base dunque su nessun valore 'finale', mentre per time_months ho deciso di escluderlo perchè non è facilmente interpretabile e ha una distribuzione troppo elevata

#Inizializzo l'encoder per trasformare le colonne categoriali
encoder = LabelEncoder()

#Eseguo la trasformazione, ottenendo prima tutte le colonne categoriali e poi mediante l'encoder eseguo la trasformazione
for col in features_.select_dtypes(include=['object']).columns:
    features_[col] = encoder.fit_transform(features_[col])


scalar = MinMaxScaler() #Utilizzo MinMaxScaler per normalizzare i dati
scalar_feauter = scalar.fit_transform(features_) #Normalizzo i dati

pca = PCA(n_components=2) #Setto PCA per ridurre la dimensionalità dei dati così da ottenere una migliora rappresentazione, setto n_components=2 per indicare le componenti da mantenere dopo la riduzione
features = pca.fit_transform(scalar_feauter) #trasformo la dimensione dei dati

K= range(2,10) #Setto il range di cluster da formare tramite k-means

#Utilizzo la tecnica del gomito per comprendere quanti cluster formare
inertia = []
for k in K: #Inizio un ciclo in cui per ogni k in K eseguo k-means con k cluster e salvo l'inertia (somma delle sitanze quadratiche tra ogni punto del cluster ed il centroide) per comprendere con quale k è più efficiente
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)
    labelss = kmeans.labels_

#Rappresento a video la curvatura dell'inerzia per ogni k testato su k-means
plt.figure(figsize=(10,6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Numeri di cluster (k)')
plt.ylabel('Inertia')
plt.title('Gomito')
plt.grid(True)
plt.show()

#Da qui ho notato che tramite k=3 si hanno i risultati migliori, dunque ho eseguito k-means con k=3 così da mostrare il cluster ottenuto
kmenas = KMeans(n_clusters=3, random_state=42 )
kmenas.fit(features) #Addestro il modello
labels = kmenas.labels_ #Ottengo le labels

features_['Cluster'] = labels #Creo una nuova colonna 'Cluster' in cui indico a quale cluster è stato associato ogni elemento

#Calcolo e stampo a video il valore delle metriche di valutazione
score_silhouette = silhouette_score(features, labels)
print(f'Punteggio Silhouette: {score_silhouette}')

score_calinski = calinski_harabasz_score(features, labels)
print(f'Punteggio Calinski: {score_calinski} ')

score_davies = davies_bouldin_score(features, labels)
print(f'Punteggio Davies: {score_davies}')



#Mostro il cluster ottenuto
plt.figure(figsize=(12, 8))
scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
centroids = kmenas.cluster_centers_ #Ottengo i centroidi dei cluster per mostrarli di rosso tramite delle X 
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-means (k=3)')
plt.xlabel('Component 1 PCA')#Rappresenta quanto il valore di ogni punto della componente 1 di PCA si discosta dalla media dei valori
plt.ylabel('Component 2 PCA')
plt.legend()
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

#Quindi nello specifico PCA assegna delle nuove coordinate ad ogni punto e tramite plot si mostra il discostamento del valore assegnato al punto dato rispetto alla media dei valori

#Calcolo delle statistiche descrittive per ciascun cluster così da comprendere la differenza presente tra i cluster
cluster_summary = features_.groupby('Cluster').mean()

# Heatmap delle medie delle caratteristiche nei diversi cluster
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary.T, annot=True, cmap='viridis')
plt.title('Heatmap')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()
