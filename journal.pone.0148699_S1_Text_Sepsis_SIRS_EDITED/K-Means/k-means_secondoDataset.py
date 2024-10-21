import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns

##Leggo il dataset dal file csv
dataset = pd.read_csv('C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv')
dataset = dataset.dropna()

print(dataset.info()) ## Da qui noto tutte le informazioni dei dati, sopratutto il tipo di dato presente così da capire se pre-elaborare i dati

#Ho notato che sono tutti di tipo float64 e int64 quindi vanno bene per l'elaborazione tramite k-means

scalar = MinMaxScaler() #Utilizzo MinMaxScaler per normalizzare i dati
scalar_feauter = scalar.fit_transform(dataset) #Normalizzo i dati

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
    score = silhouette_score(features, labelss) #Calcolo il valore Silhouette così da stamparlo
    score_calinski = calinski_harabasz_score(features, labelss)#Calcolo il valore Calinski così da stamparlo
    score_davies = davies_bouldin_score(features, labelss)#Calcolo il valore Davies così da stamparlo
    print(f'Punteggio Silhouette con {k} cluster pari a: {score}')
    print(f'Punteggio Calinski con {k} cluster pari a: {score_calinski}')
    print(f'Punteggio Davies con {k} cluster pari a: {score_davies}')

#Rappresento a video la curvatura dell'inerzia per ogni k testato su k-means
plt.figure(figsize=(10,6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Numeri di cluster (k)')
plt.ylabel('Inertia')
plt.title('Gomito')
plt.grid(True)
plt.show()

#Da qui ho notato che tramite k=4 si hanno i risultati migliori, dunque ho eseguito k-means con k=4 così da mostrare il cluster ottenuto
kmenas = KMeans(n_clusters =4, random_state=42)
kmenas.fit(features) #Addestro il modello
labels = kmenas.labels_ #Ottengo le labels

dataset['Cluster'] = labels #Creo una nuova colonna 'Cluster' in cui indico a quale cluster è stato associato ogni elemento

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
plt.title('K-means (k=4)')
plt.xlabel('Component 1 PCA')#Rappresenta quanto il valore di ogni punto della componente 1 di PCA si discosta dalla media dei valori
plt.ylabel('Component 2 PCA')
plt.legend()
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

#Stampo a video l'heatMap 
features_only = dataset.drop(columns=['Cluster'], errors='ignore') #Escludo la colonna Cluster

# Converto 'scaler_feauter' in un DataFrame Pandas con i nomi delle colonne originali
features_df = pd.DataFrame(scalar_feauter, columns=features_only.columns)

features_df['Cluster'] = dataset['Cluster'].values

cluster_summary = features_df.groupby('Cluster').mean()

plt.figure(figsize=(12,8))
sns.heatmap(cluster_summary.T, annot=True, cmap='viridis')
plt.xlabel("Cluster")
plt.ylabel("Features")
plt.title("HeatMap")
plt.show()