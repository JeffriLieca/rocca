import numpy as np
from ..clustering import KMeans

# Membuat data manual untuk 3 kluster
kluster1 = np.random.randn(50, 2) + np.array([2, 2])
kluster2 = np.random.randn(50, 2) + np.array([-2, -2])
kluster3 = np.random.randn(50, 2) + np.array([2, -2])

# Menggabungkan data ke dalam satu array
X = np.vstack((kluster1, kluster2, kluster3))

# Menginisialisasi dan melatih model KMeans
kmeans = KMeans(k=3, verbose=True,init_method='k-means++',scaling_method='minmax',distance_metric='euclidean')
kmeans.fit(X)

# Menggunakan metode silhouette dan elbow untuk menentukan jumlah kluster optimal
KMeans.silhouette_score_method(X, max_k=10)
KMeans.elbow_method(X, max_k=10)

# Visualisasi kluster tanpa dan dengan PCA
kmeans.visualize_clusters(X, use_pca=False)
kmeans.visualize_clusters(X, use_pca=True) #Pakai PCA jika fitur > 2

# Print hasil kluster dengan labels_
print(kmeans.labels_)