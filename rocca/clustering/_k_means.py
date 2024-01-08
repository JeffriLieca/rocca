
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd


class KMeans:
    """
    Implementasi algoritma pengelompokan KMeans.

    Algoritma ini mengelompokkan data menjadi sejumlah kluster yang ditentukan dan mengoptimalkan posisi sentroid kluster.

    Attributes:
        k (int): Jumlah kluster.
        max_iters (int): Maksimum iterasi yang dilakukan.
        init_method (str): Metode inisialisasi ('k-means++' atau 'random').
        tol (float): Toleransi untuk konvergensi.
        verbose (bool): Mode verbose untuk mencetak detail proses.
        scaling_method (str atau None): Metode penskalaan dataset.
        distance_metric (str): Metrik jarak ('euclidean' atau 'manhattan').
        centroids (np.ndarray): Sentroid dari setiap kluster.
        clusters (list): Daftar kluster dengan indeks anggota.
        inertia_ (float): Jumlah kuadrat jarak sampel ke sentroid terdekatnya.
        labels_ (np.ndarray): Label untuk setiap titik.
    """
    def __init__(self, k=3, max_iters=1000, init_method='k-means++', tol=1e-4, verbose=False, scaling_method=None, distance_metric='euclidean'):
        """
        Inisialisasi KMeans dengan parameter yang ditentukan.

        Args:
            k (int): Jumlah kluster.
            max_iters (int): Maksimum iterasi.
            init_method (str): Metode inisialisasi.
            tol (float): Toleransi untuk konvergensi.
            verbose (bool): Mode verbose.
            scaling_method (str atau None): Metode penskalaan dataset.
            distance_metric (str): Metrik jarak.
        """
        self.k = k
        self.max_iters = max_iters
        self.init_method = init_method
        self.tol = tol
        self.verbose = verbose
        self.scaling_method = scaling_method
        self.distance_metric = distance_metric
        self.centroids = None
        self.clusters = [[] for _ in range(self.k)]
        self.inertia_ = 0
        self.feature_names=None
        self.labels_=None

    def fit(self, X):
        """
        Melakukan pelatihan model KMeans pada dataset X.

        Proses ini mencakup penskalaan data, inisialisasi sentroid, pembentukan kluster, dan pembaruan sentroid.

        Args:
            X (np.ndarray atau pd.DataFrame): Dataset untuk pelatihan.

        Returns:
            self: Objek KMeans yang telah dilatih.
        """
        # Cek apakah penskalaan diperlukan
        if self.scaling_method is not None:
            self._trace("Melakukan penskalaan data...")
            X_scaled = self._scale_data(X, self.scaling_method)
        else:
            X_scaled = X

        self._trace("Menginisialisasi sentroid awal secara random...")
        self.centroids = self._initialize_centroids(X_scaled)
        self._trace(f"Sentroid awal: \n{self.centroids}")

        for iteration in range(self.max_iters):
            self._trace(f"\n--- Iterasi {iteration + 1}/{self.max_iters} ---")
            self.clusters = self._create_clusters(X_scaled)
            self._trace(f"Kluster terbentuk ")
            previous_centroids = self.centroids
            self.centroids = self._calculate_centroids(X_scaled)
            self._trace("Menghitung ulang sentroid tiap Kluster dengan Mean dari Kluster")
            self._trace(f"Sentroid diperbarui: \n{self.centroids}")

            diff = np.linalg.norm(self.centroids - previous_centroids)
            self._trace(f"Perubahan sentroid: {diff}")
            if diff < self.tol:
                self._trace("\nTidak ada perubahan sentroid.")
                self._trace("Konvergensi tercapai.")
                break

        self.inertia_ = self._calculate_inertia(X_scaled)
        self._trace("\nProses pelatihan selesai.")
        self.labels_ = np.empty(X_scaled.shape[0], dtype=int)
        for idx, data_point in enumerate(X_scaled):
            closest_centroid = np.argmin(np.linalg.norm(data_point - self.centroids, axis=1))
            self.labels_[idx] = closest_centroid

        return self

    def _initialize_centroids(self, X):
        """
        Menginisialisasi sentroid untuk pengelompokan KMeans.

        Inisialisasi dilakukan berdasarkan metode 'k-means++' atau 'random'.

        Args:
            X (np.ndarray): Dataset.

        Returns:
            np.ndarray: Array sentroid yang diinisialisasi.
        """
        n_samples, n_features = X.shape
        n_samples, n_features = X.shape
        if self.init_method == 'random':
            indices = np.random.choice(n_samples, self.k, replace=False)
            centroids = X[indices]
        elif self.init_method == 'k-means++':
            centroids = np.zeros((self.k, n_features))
            first_centroid_idx = np.random.choice(n_samples)
            centroids[0] = X[first_centroid_idx]
            for centroid_idx in range(1, self.k):
                distances = np.array([min([np.linalg.norm(x - centroid) ** 2 for centroid in centroids[:centroid_idx]]) for x in X])
                probabilities = distances / distances.sum()
                cumulative_probabilities = np.cumsum(probabilities)
                r = np.random.rand()
                for idx, prob in enumerate(cumulative_probabilities):
                    if r < prob:
                        centroids[centroid_idx] = X[idx]
                        break
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        
        
        return centroids
    

    def _calculate_distance(self, data_point, centroid):
        """
        Menghitung jarak antara titik data dan sentroid.

        Jarak dihitung berdasarkan metrik yang ditentukan ('euclidean' atau 'manhattan').

        Args:
            data_point (np.ndarray): Titik data.
            centroid (np.ndarray): Sentroid.

        Returns:
            float: Jarak yang dihitung.
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((data_point - centroid) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(data_point - centroid))
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def _create_clusters(self, X):
        """
        Membentuk kluster berdasarkan jarak terdekat antara data dan sentroid.

        Args:
            X (np.ndarray): Dataset.

        Returns:
            list: Daftar kluster dengan indeks anggota.
        """
        clusters = [[] for _ in range(self.k)]
        for idx, data_point in enumerate(X):
            distances = [self._calculate_distance(data_point, centroid) for centroid in self.centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(idx)
        return clusters
    
    

    def _calculate_centroids(self, X):
        """
        Menghitung ulang sentroid untuk setiap kluster.

        Sentroid dihitung sebagai rata-rata titik data dalam kluster.

        Args:
            X (np.ndarray): Dataset.

        Returns:
            np.ndarray: Array sentroid yang diperbarui.
        """
        centroids = np.zeros((self.k, X.shape[1]))
        for idx, cluster in enumerate(self.clusters):
            if len(cluster) == 0:  # Handling empty clusters
                centroids[idx] = X[np.random.choice(X.shape[0])]
            else:
                centroids[idx] = np.mean(X[cluster], axis=0)
        return centroids

    def _calculate_inertia(self, X):
        """
        Menghitung inersia total dari pengelompokan saat ini.

        Inersia dihitung sebagai jumlah jarak kuadrat dari sampel ke sentroid terdekatnya.

        Args:
            X (np.ndarray): Dataset.

        Returns:
            float: Nilai inersia.
        """
        inertia = 0
        for idx, cluster in enumerate(self.clusters):
            inertia += np.sum(np.linalg.norm(X[cluster] - self.centroids[idx], axis=1) ** 2)
        return inertia

    def predict(self, X):
        """
        Memprediksi kluster terdekat untuk setiap titik dalam dataset X.

        Args:
            X (np.ndarray): Dataset untuk prediksi.

        Returns:
            list: Daftar prediksi kluster.
        """
        return [np.argmin(np.linalg.norm(x - self.centroids, axis=1)) for x in X]

   
    
    def get_cluster_labels(self):
        """
        Mengembalikan label kluster untuk setiap sampel yang digunakan dalam fit.

        Returns:
            np.array: Array yang berisi label kluster untuk setiap sampel.
        """
        return self.labels_

    
   
    def _pca_reduce(self, X, n_components=2):
        """
        Mengurangi dimensi dataset menggunakan PCA untuk visualisasi.

        Args:
            X (np.ndarray): Dataset.
            n_components (int): Jumlah komponen PCA.

        Returns:
            tuple: Dataset yang telah ditransformasi dan matriks transformasi.
        """
        X_centered = X - np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        W2 = Vt.T[:, :n_components]
        X_pca = X_centered.dot(W2)
        return X_pca, W2  

    def visualize_clusters(self, X, use_pca=False):
        """
        Memvisualisasikan kluster menggunakan scatter plot.

        Args:
            X (np.ndarray): Dataset.
            use_pca (bool): Apakah menggunakan PCA untuk reduksi dimensi.

        Returns:
            None: Fungsi menghasilkan plot namun tidak mengembalikan nilai.
        """
        if self.scaling_method:
            X_scaled = self._scale_data(X, self.scaling_method)
        else:
            X_scaled = X

        # Aplikasikan PCA jika diminta oleh pengguna
        if use_pca:
            X_reduced, pca_transform = self._pca_reduce(X_scaled)
            centroids_reduced = self._pca_reduce(self.centroids, n_components=pca_transform.shape[1])[0]
        else:
            X_reduced = X_scaled
            centroids_reduced=self.centroids
        

        # Visualize the clusters
        plt.figure(figsize=(12, 6))
        for i, points in enumerate(self.clusters):
            cluster_points = X_reduced[points]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.5, label=f'Cluster {i+1}')
        plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], color='black', marker='x', s=100, label='Centroids')

        x_offset = (X_reduced[:, 0].max() - X_reduced[:, 0].min()) * 0.02  # 2% of the range
        y_offset = (X_reduced[:, 1].max() - X_reduced[:, 1].min()) * 0.02  # 2% of the range

        for i, centroid in enumerate(centroids_reduced):
            plt.text(centroid[0] + x_offset, centroid[1] + y_offset, f'C{i+1}', fontsize=12, color='red', ha='center', va='center')

        title = 'Cluster Visualization' + (' in 2D using PCA' if use_pca else '')
        plt.title(title)
        plt.legend()
        plt.show()




    def _trace(self, message):
        """
        Mencetak pesan jika mode verbose diaktifkan.

        Args:
            message (str): Pesan yang akan dicetak.

        Returns:
            None: Fungsi untuk mencetak pesan dan tidak mengembalikan nilai.
        """
        if self.verbose:
            print(message)



    @staticmethod
    def _scale_data(X, method='standard'):
        """
        Menyesuaikan skala data menggunakan metode yang ditentukan.

        Args:
            X (np.ndarray): Dataset yang akan disesuaikan skalanya.
            method (str): Metode penskalaan ('standard', 'minmax', 'maxabs', 'robust').

        Returns:
            np.ndarray: Dataset dengan skala yang telah disesuaikan.
        """
        if method == 'standard':
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            return (X - means) / stds
        elif method == 'minmax':
            mins = np.min(X, axis=0)
            maxs = np.max(X, axis=0)
            return (X - mins) / (maxs - mins)
        elif method == 'maxabs':
            max_abs_vals = np.max(np.abs(X), axis=0)
            return X / max_abs_vals
        elif method == 'robust':
            median = np.median(X, axis=0)
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            return (X - median) / (q3 - q1)
        else:
            raise ValueError("Unknown scaling method: {}".format(method))



    @staticmethod
    def silhouette_score_method(X, max_k):
        """
        Menghitung skor silhouette untuk berbagai jumlah kluster dan menentukan k optimal.

        Args:
            X (np.ndarray): Dataset.
            max_k (int): Maksimum jumlah kluster yang akan diuji.

        Returns:
            None: Fungsi menghasilkan plot namun tidak mengembalikan nilai.
        """

        silhouette_avg_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(k=k)
            kmeans.fit(X)
            labels = kmeans.get_cluster_labels()
            silhouette_avg = silhouette_score(X, labels)
            silhouette_avg_scores.append(silhouette_avg)

        optimal_k_index = silhouette_avg_scores.index(max(silhouette_avg_scores))
        optimal_k = range(2, max_k + 1)[optimal_k_index]

        plt.figure(figsize=(8, 4))
        plt.plot(range(2, max_k + 1), silhouette_avg_scores, marker='o')

        plt.scatter(optimal_k, silhouette_avg_scores[optimal_k_index], color='red', edgecolor='black', zorder=5, s=100)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Average Silhouette Score')
        plt.title('Silhouette Scores for Different k Values')
        plt.show()




    @staticmethod
    def elbow_method(X, max_k):
        """
        Menggunakan metode Elbow untuk menentukan jumlah kluster optimal.

        Args:
            X (np.ndarray): Dataset.
            max_k (int): Maksimum jumlah kluster yang akan diuji.

        Returns:
            None: Fungsi menghasilkan plot namun tidak mengembalikan nilai.
        """

        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(k=k)
            kmeans.fit(X)  # Use scaled data
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, max_k + 1), inertias, 'o-')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters, k')
        plt.ylabel('Inertia')
        plt.show()

