from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering, Birch, MeanShift
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances_argmin_min
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

dataset_path = 'C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\journal.pone.0175818_S1Dataset_Spain_cardiac_arrest_EDITED..csv'
dataset = pd.read_csv(dataset_path)
dataset = dataset.dropna()
algorithm = 'agglomerative'

scaler = MinMaxScaler()#Utilizzo MinMaxScaler per normalizzare i dati
    
if '10_7717_peerj_5665_dataYM2018_neuroblastoma' in dataset_path:
    features_ = dataset.drop(columns=['time_months', 'outcome'])#Escludo outcome perchè pongo il clustering sui valori di base dunque su nessun valore 'finale', mentre per time_months ho deciso di escluderlo perchè non è facilmente interpretabile e ha una distribuzione troppo elevata
    #Inizializzo l'encoder per trasformare le colonne categoriali
    encoder = LabelEncoder()
    #Eseguo la trasformazione, ottenendo prima tutte le colonne categoriali e poi mediante l'encoder eseguo la trasformazione
    for col in features_.select_dtypes(include=['object']).columns:
        features_[col] = encoder.fit_transform(features_[col])
    
    if 'k-means' in algorithm:
        clusters=3 #Indica il numero di cluster
    elif 'birch' in algorithm:
        clusters=3 #Indica il numero di cluster
        t= 0.2 #Indica il threshold
    elif 'dbscan' in algorithm:
        e = 0.5 #Indica eps
    elif 'agglomerative' in algorithm:
        clusters=4 #Indica il numero di cluster
        link = 'average' #Indica linkage
        aff = 'manhattan' #Indica affinity
    elif 'spectral' in algorithm:
        clusters = 3 #Indica il numero di cluster
        aff = 'rbf' #Indica affinity
        gam = 10 #Indica gamma
        neighbors = 0 #Indica n_neighbors
    elif 'mean-shift' in algorithm:
        band = 1 #Indica bandwidth
        seeding = False

elif 'journal.pone.0158570_S2File_depression_heart_failure' in dataset_path:
    features_ = dataset.drop(columns=['id'])
    scaler = StandardScaler() #Utilizzo StandardScaler per normalizzare i dati

    if 'k-means' in algorithm:
        clusters=3 #Indica il numero di cluster
    elif 'birch' in algorithm:
        clusters=None #Indica il numero di cluster
        t= 1.4 #Indica il threshold
    elif 'dbscan' in algorithm:
        e = 0.9 #Indica eps
    elif 'agglomerative' in algorithm:
        clusters=3 #Indica il numero di cluster
        link = 'ward' #Indica linkage
        aff = 'euclidean'
    elif 'spectral' in algorithm:
        clusters = 3 #Indica il numero di cluster
        aff = 'nearest_neighbors' #Indica affinity
        gam = 0 #Indica gamma
        neighbors = 40 
    elif 'mean-shift' in algorithm:
        band = 1.5 #Indica bandwidth
        seeding = True

elif 'journal.pone.0175818_S1Dataset_Spain_cardiac_arrest_EDITED.' in dataset_path:
    features_ = dataset

    if 'k-means' in algorithm:
        clusters=4 #Indica il numero di cluster
    elif 'birch' in algorithm:
        clusters=None #Indica il numero di cluster
        t= 0.5 #Indica il threshold
    elif 'dbscan' in algorithm:
        e = 0.3 #Indica eps
    elif 'agglomerative' in algorithm:
        clusters=4 #Indica il numero di cluster
        link = 'ward' #Indica linkage
        aff = 'euclidean'
    elif 'spectral' in algorithm:
        clusters = 4 #Indica il numero di cluster
        aff = 'nearest_neighbors' #Indica affinity
        gam = 0 #Indica gamma
        neighbors = 130
    elif 'mean-shift' in algorithm:
        band = 0.45  #Indica bandwidth
        seeding = False #Indica bin_seeding

elif 'journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED' in dataset_path:
    features_ = dataset

    if 'k-means' in algorithm:
        clusters=4 #Indica il numero di cluster
    elif 'birch' in algorithm:
        clusters=None #Indica il numero di cluster
        t= 0.3 #Indica il threshold
    elif 'dbscan' in algorithm:
        e = 0.5 #Indica eps
    elif 'agglomerative' in algorithm:
        clusters=4 #Indica il numero di cluster
        link = 'ward' #Indica linkage
        aff = 'euclidean'
    elif 'spectral' in algorithm:
        clusters = 4 #Indica il numero di cluster
        aff = 'rbf' #Indica affinity
        gam =  35#Indica gamma
        neighbors = 0
    elif 'mean-shift' in algorithm:
        band = None  #Indica bandwidth
        seeding = True #Indica bin_seeding

elif 'Takashi2019_diabetes_type1_dataset_preprocessed' in dataset_path:
    features_ = dataset

    if 'k-means' in algorithm:
        clusters=3 #Indica il numero di cluster
    elif 'birch' in algorithm:
        clusters=None #Indica il numero di cluster
        t= 0.52 #Indica il threshold
    elif 'dbscan' in algorithm:
        e = 0.3 #Indica eps
    elif 'agglomerative' in algorithm:
        clusters=3 #Indica il numero di cluster
        link = 'ward' #Indica linkage
        aff = 'euclidean'
    elif 'spectral' in algorithm:
        clusters = 3 #Indica il numero di cluster
        aff = 'rbf' #Indica affinity
        gam = 30 #Indica gamma
        neighbors = 0 #Indica n_neighbors
    elif 'mean-shift' in algorithm:
        band = 1.35  #Indica bandwidth
        seeding = False #Indica bin_seeding

scaler_feauter = scaler.fit_transform(features_) #Normalizzo i dati

if(dataset_path == "C:\\Users\\fncba\\OneDrive\Documenti\\Stage\\Cartelle cliniche\\Takashi2019_diabetes_type1_dataset_preprocessed.csv" and algorithm=="mean-shift"):
    pca = PCA() #Non imposto n_comonents poichè l'algortimo richiede la struttura originale dei dati per funzionare correttamente 
else:
    pca = PCA(n_components=2) #Setto PCA per ridurre la dimensionalità dei dati così da ottenere una migliora rappresentazione, setto n_components=2 per indicare le componenti da mantenere dopo la riduzione
features = pca.fit_transform(scaler_feauter) #trasformo la dimensione dei dati
    
if 'k-means' in algorithm:
    start_time = time.time()
    K= range(2,10) #Setto il range di cluster da formare tramite k-means

    #Utilizzo la tecnica del gomito per comprendere quanti cluster formare
    inertia = []
    for k in K: #Inizio un ciclo in cui per ogni k in K eseguo k-means con k cluster e salvo l'inertia (somma delle sitanze quadratiche tra ogni punto del cluster ed il centroide) per comprendere con quale k è più efficiente
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
        labelss = kmeans.labels_


    kmenas = KMeans(n_clusters=clusters, random_state=42 )
    kmenas.fit(features) #Addestro il modello
    labels = kmenas.labels_ #Ottengo le labels

    end_time = time.time()
    
    features_['Cluster'] = labels #Creo una nuova colonna 'Cluster' in cui indico a quale cluster è stato associato ogni elemento

    # Conto il numero di outlier presenti nel clustering
    centroids = kmeans.cluster_centers_ #Ottengo i centroidi

    closest,distanze = pairwise_distances_argmin_min(features, centroids) #Calcolo la distanza e le salvo nell'array distanze. Inoltre, closest è l'array che contiene il centroide più vicino per un determinato punto

    soglia = np.percentile(distanze, 95) #Uso il 95-esimo quartile delle distanze come soglia

    outliers = np.where(distanze > soglia)[0] #Ottengo dalla tupla solo le distanze che sono maggiori della soglia

    outliers_number = len(outliers) #Conto il numero di errori

    #Stampo il numero di outlier
    print(f"Numero di outlier: {outliers_number}")
    #Rappresento a video la curvatura dell'inerzia per ogni k testato su k-means
    plt.figure(figsize=(10,6))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Numeri di cluster')
    plt.ylabel('Inertia')
    plt.title('Tecnica del Gomito')
    plt.grid(True)
    plt.show()

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

elif 'spectral' in algorithm:
    start_time = time.time()
    inertia=[]
    K=range(2,10)

    for k in K:
        spectral = SpectralClustering(n_clusters=k, affinity=aff, random_state=42)
        spectral.fit(features)
        labels = spectral.labels_
        centroids = np.array([features[labels == i].mean(axis=0) for i in range(k)])

        # Calcolo della "inerzia" come somma delle distanze quadrate dai centroidi
        closest, distances = pairwise_distances_argmin_min(features, centroids)
        inertia_ = np.sum(distances ** 2)
        inertia.append(inertia_)

    spectral = SpectralClustering(n_clusters=clusters, affinity=aff, gamma=gam, random_state=42, n_neighbors= neighbors)
    spectral.fit(features)

    end_time = time.time()

    labels = spectral.labels_
    features_['Cluster'] = labels


    # Conto il numero di outlier presenti nel clustering
    centroids = np.array([features[labels == i].mean(axis=0) for i in np.unique(labels)]) #Ottengo i centroidi manualmente tramite la media dei punti presenti in essi

    closest,distanze = pairwise_distances_argmin_min(features, centroids) #Calcolo la distanza e le salvo nell'array distanze. Inoltre, closest è l'array che contiene il centroide più vicino per un determinato punto

    soglia = np.percentile(distanze, 95) #Uso il 95-esimo quartile delle distanze come soglia

    outliers = np.where(distanze > soglia)[0] #Ottengo dalla tupla solo le distanze che sono maggiori della soglia

    outliers_number = len(outliers) #Conto il numero di errori

    #Stampo il numero di outlier
    print(f"Numero di outlier: {outliers_number}")
    #Rappresento a video la curvatura dell'inerzia per ogni k testato su spectral
    plt.figure(figsize=(10,6))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Numeri di cluster')
    plt.ylabel('Inertia')
    plt.title('Gomito')
    plt.grid(True)
    plt.show()

    #Rappresento a video i cluster ottenuti
    plt.figure(figsize=(12,8))
    plt.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
    plt.xlabel("Component 1 PCA")
    plt.ylabel("Component 2 PCA")
    plt.grid(True)
    plt.title("Spectral Clustering")
    plt.show()

elif 'birch' in algorithm:
    start_time = time.time()
    birch = Birch(threshold=t, n_clusters=clusters) #Ho provato diversi valori per branching_factor ma non cambia nulla, mentre per threshold ho notato che impostandolo a 0.2 si ha un miglior clustering. Inoltre ho settato n_cluster=3 (default) poichè settato a None vengono generati 18 cluster
    birch.fit(features)

    end_time=time.time()
    
    labels = birch.labels_
    features_['Cluster'] = labels

    # Conto il numero di outlier presenti nel clustering
    centroids = birch.subcluster_centers_ #Ottengo i centroidi

    closest,distanze = pairwise_distances_argmin_min(features, centroids) #Calcolo la distanza e le salvo nell'array distanze. Inoltre, closest è l'array che contiene il centroide più vicino per un determinato punto

    soglia = np.percentile(distanze, 95) #Uso il 95-esimo quartile delle distanze come soglia

    outliers = np.where(distanze > soglia)[0] #Ottengo dalla tupla solo le distanze che sono maggiori della soglia

    outliers_number = len(outliers) #Conto il numero di errori

    #Stampo il numero di outlier
    print(f"Numero di outlier: {outliers_number}")

    plt.figure(figsize=(12,8))
    plt.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
    plt.title("BIRCH Clustering")
    plt.xlabel("Component 1 PCA")
    plt.ylabel("Component 2 PCA")
    plt.grid(True)
    plt.show()

elif 'dbscan' in algorithm:
    start_time = time.time()
    dbscan = DBSCAN(eps=e) 
    dbscan.fit(features)

    end_time = time.time()
   
    labels = dbscan.labels_

    num = np.sum(labels==-1)
    features_['Cluster'] = labels

    print(f"Numero errori:{num}")

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Component 1 PCA')
    plt.ylabel('Component 2 PCA')
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True)
    plt.show()

elif 'agglomerative' in algorithm:
    start_time = time.time()

    #specifico il criterio utilizzato per unire i cluster durante il processo di clustering gerarchico e salvo il collegamento in Z
    Z = linkage(features, link)

    #Eseguo il cluster con k=4
    agglomerative = AgglomerativeClustering(n_clusters=clusters, linkage=link, metric=aff)
    agglomerative.fit(features)

    end_time = time.time()

    labels= agglomerative.labels_

    features_['Cluster'] = labels

    # Conto il numero di outlier presenti nel clustering

    centroids = np.array([features[labels == i].mean(axis=0) for i in np.unique(labels)]) #Calcolo manualmente i centroidi come la media dei punti presenti in esso

    closest,distanze = pairwise_distances_argmin_min(features, centroids) #Calcolo la distanza e le salvo nell'array distanze. Inoltre, closest è l'array che contiene il centroide più vicino per un determinato punto

    soglia = np.percentile(distanze, 95) #Uso il 95-esimo quartile delle distanze come soglia

    outliers = np.where(distanze > soglia)[0] #Ottengo dalla tupla solo le distanze che sono maggiori della soglia

    outliers_number = len(outliers) #Conto il numero di errori

    #Stampo il numero di outlier
    print(f"Numero di outlier: {outliers_number}")

    #Mostro a video il dendrogramma per comprendere il numero di cluster da formare (k=4)
    plt.figure(figsize=(10,7))
    dendrogram(Z)
    plt.title('Dendrogramma')
    plt.xlabel('Cluster')
    plt.ylabel('Distance')
    plt.show()

    #Mostro il clustering ottenuto
    plt.figure(figsize=(12,8))
    plt.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
    plt.title("Agglomerative Clustering")
    plt.xlabel("Component 1 PCA")
    plt.ylabel("Component 2 PCA")
    plt.grid(True)
    plt.show()

elif 'mean-shift' in algorithm:
    start_time = time.time()

    mean = MeanShift(bandwidth=band, bin_seeding= seeding)
    mean.fit(features)

    end_time = time.time()

    labels = mean.labels_
    features_['Cluster'] = labels

    # Conto il numero di outlier presenti nel clustering
    centroids = mean.cluster_centers_ #Ottengo i centroidi

    closest,distanze = pairwise_distances_argmin_min(features, centroids) #Calcolo la distanza e le salvo nell'array distanze. Inoltre, closest è l'array che contiene il centroide più vicino per un determinato punto

    soglia = np.percentile(distanze, 95) #Uso il 95-esimo quartile delle distanze come soglia

    outliers = np.where(distanze > 3)[0] #Ottengo dalla tupla solo le distanze che sono maggiori della soglia

    outliers_number = len(outliers) #Conto il numero di errori

    #Stampo il numero di outlier
    print(f"Numero di outlier: {outliers_number}")

    plt.figure(figsize=(12,8))
    plt.scatter(features[:,0], features[:,1], c=labels, cmap='viridis')
    plt.xlabel("Component 1 PCA")
    plt.ylabel("Component 2 PCA")
    plt.grid(True)
    plt.title("Mean-Shift Clustering")
    plt.show()

#Calcolo e stampo a video il valore delle metriche di valutazione
score_silhouette = silhouette_score(features, labels)
print(f'Punteggio Silhouette: {score_silhouette}')

score_calinski = calinski_harabasz_score(features, labels)
print(f'Punteggio Calinski: {score_calinski} ')

score_davies = davies_bouldin_score(features, labels)
print(f'Punteggio Davies: {score_davies}')

#Calcolo delle statistiche descrittive per ciascun cluster così da comprendere la differenza presente tra i cluster
features_only = features_.drop(columns=['Cluster'], errors='ignore') #Escludo la colonna Cluster

# Converto 'scaler_feauter' in un DataFrame Pandas con i nomi delle colonne originali
features_df = pd.DataFrame(scaler_feauter, columns=features_only.columns)

features_df['Cluster'] = features_['Cluster'].values

cluster_summary = features_df.groupby('Cluster').mean()

# Heatmap delle medie delle caratteristiche nei diversi cluster
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary.T, annot=True, cmap='viridis')
plt.title('Heatmap')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.subplots_adjust(left=0.3, right=0.8)
plt.show()


total_time = (end_time - start_time) * 1000
print(f"Tempo impiegato: {total_time} " )
