import unittest
import numpy as np
from rocca.clustering import KMeans  

class TestKMeans(unittest.TestCase):

    def setUp(self):
        # Membuat dataset sederhana
        self.kluster1 = np.random.randn(50, 2) + np.array([2, 2])
        self.kluster2 = np.random.randn(50, 2) + np.array([-2, -2])
        self.kluster3 = np.random.randn(50, 2) + np.array([2, -2])

        # Menggabungkan data ke dalam satu array
        self.X = np.vstack((self.kluster1, self.kluster2, self.kluster3))
        self.kmeans = KMeans(k=3, init_method='k-means++', scaling_method='minmax', distance_metric='euclidean')

    def test_fit(self):
        # Menguji apakah metode fit berjalan tanpa error
        self.kmeans.fit(self.X)
        self.assertIsNotNone(self.kmeans.centroids, "Centroids tidak boleh None setelah fitting.")
        
    def test_predict(self):
        # Menguji apakah metode predict mengembalikan output yang benar
        self.kmeans.fit(self.X)
        predictions = self.kmeans.predict(self.X)
        self.assertEqual(len(predictions), self.X.shape[0], "Panjang prediksi harus sama dengan jumlah sampel")
        
    def test_inertia(self):
        # Menguji apakah inersia dihitung dengan benar
        self.kmeans.fit(self.X)
        self.assertGreater(self.kmeans.inertia_, 0, "Inersia harus lebih besar dari 0 setelah fitting.")
        
    def test_labels_(self):
        # Menguji apakah labels_ dihitung dengan benar
        self.kmeans.fit(self.X)
        unique_labels = set(self.kmeans.labels_)
        self.assertEqual(len(unique_labels), self.kmeans.k, "Jumlah label unik harus sama dengan jumlah kluster")

# Menjalankan test jika file ini dijalankan sebagai script
if __name__ == '__main__':
    unittest.main()
