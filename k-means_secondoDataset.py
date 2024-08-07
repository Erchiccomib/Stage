import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns

##Leggo il dataset dal file csv
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv')

dataset = dataset.dropna()

print(dataset.info()) ## Da qui noto che non tutti i valori presenti sono di tipo float64

features_ = dataset.values #Ottengo i valori di ogni colonna

encoder = OneHotEncoder(sparse=False) #Setto l'encoder impostando sparse=False che permette una restituzione di una matrice densa rispetto che sparsa 
features_object = dataset.select_dtypes(include='object') ##Ottengo tutti i valori di tipo object
features_encoded = encoder.fit_transform(features_object) ## Trasformo i valori da tipo object a tipo float64
features_numeric = dataset.select_dtypes(include='float64') ##Ottengo tutti i valori di tipo float64

columnsObject_name = encoder.get_feature_names_out(features_object.columns)

features_encoded_df = pd.DataFrame(features_encoded, columns=columnsObject_name)
features_numeric_df = pd.DataFrame(features_numeric.reset_index(drop=True))
print(features_encoded_df.columns)

features_all_numeric = np.concatenate([features_encoded_df, features_numeric_df], axis=1) ##Concateno utilizzando axis=1 che permette una concatenazione delle righe rispetto le colonne (axis=0)

features_all_numericc = pd.DataFrame(features_all_numeric)
print(features_all_numericc.info())

scalar = MinMaxScaler()
scalar_feauter = scalar.fit_transform(dataset)

pca = PCA(n_components=2) ##Setto PCA per ridurre la dimensionalità dei dati così da ottenere una migliora rappresentazione, setto n_components=2 per indicare le componenti da mantenere dopo la riduzione
features = pca.fit_transform(scalar_feauter) ##trasformo la dimensione dei dati

features_df = pd.DataFrame(features)
print(features_df.info())

K= range(2,10) ##Setto il range di cluster da formare tramite k-means

inertia = []
for k in K: ##Inizio un ciclo in cui per ogni k in K eseguo k-means con k cluster e salvo l'inertia (somma delle sitanze quadratiche tra ogni punto del cluster ed il centroide) per comprendere con quale k è più efficiente
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)
    labelss = kmeans.labels_
    score = silhouette_score(features, labelss) ##Calcolo il valore Silhouette così da stamparlo
    score_calinski = calinski_harabasz_score(features, labelss)##Calcolo il valore Calinski così da stamparlo
    score_davies = davies_bouldin_score(features, labelss)##Calcolo il valore Davies così da stamparlo
    print(f'Punteggio Silhouette con {k} cluster pari a: {score}')
    print(f'Punteggio Calinski con {k} cluster pari a: {score_calinski}')
    print(f'Punteggio Davies con {k} cluster pari a: {score_davies}')

##Rappresento a video la curvatura dell'inerzia per ogni k testato su k-means
plt.figure(figsize=(10,6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Numeri di cluster (k)')
plt.ylabel('Inertia')
plt.title('Gomito')
plt.grid(True)
plt.show()

##Da qui ho notato che tramite k=5 si hanno i risultati migliori, dunque ho eseguito k-means con k=5 così da mostrare il cluster ottenuto
kmenas = KMeans(n_clusters =4, random_state=42)
kmenas.fit(features)
labels = kmenas.labels_

dataset['Cluster'] = labels

score_silhouette = silhouette_score(features, labels)
print(f'Punteggio Silhouette: {score_silhouette}')

score_calinski = calinski_harabasz_score(features, labels)
print(f'Punteggio Calinski: {score_calinski} ')

score_davies = davies_bouldin_score(features, labels)
print(f'Punteggio Davies: {score_davies}')

##Mostro il cluster ottenuto
plt.figure(figsize=(12, 8))
scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
centroids = kmenas.cluster_centers_ ##Ottengo i centroidi dei cluster per mostrarli di rosso tramite delle X 
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-means (k=5)')
plt.xlabel('Componente 1 PCA')##Rappresenta quanto il valore di ogni punto della componente 1 di PCA si discosta dalla media dei valori
plt.ylabel('Componente 2 PCA')
plt.legend()
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

##Quindi PCA assegna delle nuove coordinate ad ogni punto e tramite plot si mostra il discostamento del valore assegnato al punto dato rispetto alla media dei valori

# Calcolo delle statistiche descrittive per ciascun cluster

cluster_summary = dataset.groupby('Cluster').mean()

# Heatmap delle medie delle caratteristiche nei diversi cluster
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary.T, annot=True, cmap='viridis')
plt.title('Heatmap of Feature Means by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()
