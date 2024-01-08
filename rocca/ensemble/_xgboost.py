
import numpy as np
from rocca.tree import DecisionTreeXGB
from rocca.utils import sigmoid, calculate_log_odds, log_loss


class XGBoost:
    """
    Implementasi XGBoost untuk klasifikasi biner.

    XGBoost adalah model ensemble yang menggabungkan beberapa pohon keputusan secara berurutan. 
    Setiap pohon baru dalam ensemble berusaha memperbaiki kesalahan yang dibuat oleh model gabungan sebelumnya, 
    berbeda dengan metode ensemble paralel seperti Random Forest atau teknik agregatif seperti voting.

    Attributes:
        max_depth (int): Kedalaman maksimum setiap pohon keputusan.
        learning_rate (float): Tingkat pembelajaran untuk penyesuaian bobot pohon dalam ensemble.
        n_estimators (int): Jumlah pohon keputusan dalam ensemble.
        lambda_reg (float): Parameter regularisasi lambda.
        gamma (float): Parameter minimum loss reduction untuk membuat split baru.
        early_stopping_rounds (int, optional): Jumlah iterasi tanpa peningkatan untuk berhenti lebih awal. Default 0.
        feature_encoders (dict): Encoder untuk fitur kategorikal.
        target_encoder (dict): Encoder untuk target kategorikal.
        verbose (bool): Jika True, mencetak informasi detail selama proses fitting.
        min_child_weight (int): Bobot minimum anak untuk membuat split baru.
    """
    def __init__(self, max_depth, learning_rate, n_estimators, lambda_reg, gamma, early_stopping_rounds=0,feature_encoders={}, target_encoder=None,verbose=False,min_child_weight=0):
        """
        Inisialisasi model XGBoost.

        Args:
            max_depth (int): Kedalaman maksimum setiap pohon keputusan.
            learning_rate (float): Tingkat pembelajaran untuk penyesuaian setiap pohon.
            n_estimators (int): Jumlah pohon yang akan dibuat dalam ensemble.
            lambda_reg (float): Parameter regularisasi lambda.
            gamma (float): Minimum loss reduction yang diperlukan untuk membuat split.
            early_stopping_rounds (int, optional): Jumlah iterasi tanpa peningkatan untuk menghentikan pelatihan lebih awal. Default 0.
            feature_encoders (dict, optional): Encoder untuk fitur kategorikal. Default kosong.
            target_encoder (dict, optional): Encoder untuk target kategorikal. Default kosong.
            verbose (bool, optional): Mencetak informasi detail selama proses fitting. Default False.
            min_child_weight (int, optional): Minimum jumlah hessian yang diperlukan dalam node. Default 0.
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.trees = []
        self.best_obj = np.inf
        self.early_stopping_rounds = early_stopping_rounds
        self.best_iter = 0
        self.log_odds = None
        self.feature_encoders = feature_encoders
        self.target_encoder = target_encoder
        self.verbose=verbose
        self.min_child_weight=min_child_weight
    
    def _encode_data(self, X, y):
        """
        Mengenkripsi fitur dan target dari data training.

        Fungsi ini mengubah fitur dan target kategorikal menjadi bentuk numerik
        menggunakan pemetaan yang telah ditentukan dalam `feature_encoders` dan
        `target_encoder`.

        Args:
            X: DataFrame yang berisi fitur.
            y: Series yang berisi target.
        """
        # Tambahkan encoder hanya untuk fitur yang belum memiliki encoder
        for col in X.columns:
            if X[col].dtype == object and col not in self.feature_encoders:
                self.feature_encoders[col] = self._create_encoder(X[col])

        # Encode target jika belum dienkripsi
        if self.target_encoder is None and y.dtype == object:
            self.target_encoder = self._create_encoder(y)


    

    def _create_encoder(self, series):
        """
        Membuat encoder untuk fitur atau target kategorikal.

        Fungsi ini menghasilkan pemetaan dari nilai kategorikal ke representasi numerik.

        Args:
            series: Series yang berisi nilai kategorikal.

        Returns:
            Kamus yang memetakan nilai kategorikal ke representasi numerik.
        """
        return {val: idx for idx, val in enumerate(sorted(set(series)))}


    def _encode_column(self, series, encoder):
        """
        Mengenkripsi satu kolom menggunakan pemetaan yang diberikan.

        Fungsi ini mengubah nilai dalam kolom menjadi bentuk numerik berdasarkan
        pemetaan yang ada dalam `encoder`.

        Args:
            series: Series yang akan dienkripsi.
            encoder: Kamus yang memetakan nilai asli ke representasi numerik.

        Returns:
            Series yang telah dienkripsi.
        """
        return series.map(encoder)
    
    def _trace(self, message):
        """
        Mencetak pesan jika mode verbose diaktifkan.

        Fungsi ini digunakan untuk mencetak pesan proses atau debugging
        selama pelatihan dan prediksi jika verbose diatur ke True.

        Args:
            message: Pesan yang akan dicetak.
        """
        if self.verbose:
            print(message)

    def fit(self, X, y):
        """
        Melatih model XGBoost untuk klasifikasi biner.

        Metode ini menggunakan ensemble dari `DecisionTreeXGB` yang dioptimalkan dengan gradien 
        dan hessian dari loss function. Setiap pohon baru bertujuan untuk memperbaiki kesalahan 
        yang dibuat oleh ensemble sebelumnya.

        Args:
            X (DataFrame): Fitur training.
            y (Series): Target training (biner).
        """
        self._encode_data(X, y)

        X = X.apply(lambda col: self._encode_column(col, self.feature_encoders[col.name]) if col.name in self.feature_encoders and col.dtype == object else col)
        y = self._encode_column(y, self.target_encoder) if self.target_encoder and y.dtype == object else y
        log_odds = calculate_log_odds(y)
        self.log_odds = log_odds
        F = np.full(len(X), log_odds)
        old_loss = log_loss(y, sigmoid(F))
        self._trace(f"Iterasi ke-{0}:")
        p = sigmoid(F)
        g = p - y
        h = p * (1 - p)
        self._trace(f"  - Prediksi asli: {[round(val, 2) for val in np.array(y)[:5]]}")
        self._trace(f"  - Prediksi awal: {[round(val, 2) for val in np.array(p)[:5]]}")
        self._trace(f"  - Gradien: {[round(val, 2) for val in np.array(g)[:5]]}")
        self._trace(f"  - Hessian: {[round(val, 2) for val in np.array(h)[:5]]}")
        self._trace("")
        obj_value=0
        for i in range(self.n_estimators):
            p = sigmoid(F)
            g = p - y
            h = p * (1 - p)
            self._trace(f"Iterasi ke-{i+1}:")
            tree = DecisionTreeXGB(self.max_depth, self.lambda_reg, gamma=self.gamma,auto_encode=False,feature_encoders=self.feature_encoders,target_encoder=self.target_encoder,verbose=self.verbose,min_child_weight=self.min_child_weight)
            tree.fit(X, y, g, h, feature_encoders=self.feature_encoders, target_encoder=self.target_encoder, depth=0)
            
            if tree.actual_max_depth == 0:
                self.best_iter = i - 1
                break
            self.trees.append(tree)

            
            tree_pred = tree.predict(X,decode=False)
            F += self.learning_rate * tree_pred
            new_loss = log_loss(y, sigmoid(F))

            
            obj_value_sebelum=obj_value
            p = sigmoid(F)
            g = p - y
            h = p * (1 - p)
            obj_value = self._calculate_obj(tree)
            
            self._trace("")
            self._trace("--- XGBoost Model Information")
            self._trace(f"  - Prediksi asli : {[round(val, 2) for val in np.array(y)[:5]]}")
            self._trace(f"  - Prediksi model: {[round(val, 2) for val in np.array(p)[:5]]}")
            self._trace(f"  - Gradien       : {[round(val, 2) for val in np.array(g)[:5]]}")
            self._trace(f"  - Hessian       : {[round(val, 2) for val in np.array(h)[:5]]}")

            if obj_value < self.best_obj:
                self.best_obj = obj_value
                self.best_iter = i
            self._trace(f"  - Old Log loss      : {round(old_loss,2)}")
            self._trace(f"  - New Log loss      : {round(new_loss,2)}")
            self._trace(f"  - Old Objektif      : {round(obj_value_sebelum, 2)}")
            self._trace(f"  - New Objektif      : {round(obj_value, 2)}")
            self._trace(f"  - Best Objektif     : {round(self.best_obj, 2)}")

            if self.verbose:
                tree.title(f"Decision Tree ke-{i+1}")
                tree.visualize_tree()

            if new_loss < old_loss:
                old_loss = new_loss
                self.best_iter=i
            else:
                print("Model berhenti mengalami penurunan log loss")

            if self.early_stopping_rounds > 0:
                if i - self.best_iter >= self.early_stopping_rounds:
                    # Akan terjadi Early Stopping jika penurunan log loss dan penurunan obj value terjadi sebanyak early stopping round
                    self._trace(f"Early stopping diterapkan pada iterasi ke-{i + 1}")
                    break
            self.trees.append(tree)

    def predict(self, X,threshold=0.5,decode=True):
        """
        Membuat prediksi klasifikasi biner menggunakan model XGBoost yang telah dilatih.

        Prediksi dibuat dengan menggabungkan prediksi dari semua pohon dalam ensemble.
        Prediksi dari setiap pohon dijumlahkan, dan fungsi sigmoid digunakan untuk
        menghasilkan probabilitas kelas.

        Args:
            X (DataFrame): Fitur untuk prediksi.
            threshold (float, optional): Ambang batas untuk mengklasifikasikan output sebagai 1. Default 0.5.
            decode (bool, optional): Mendekode prediksi ke label asli jika True. Default True.

        Returns:
            numpy.array: Prediksi untuk setiap sampel dalam X.
        """
        F = np.full(len(X), self.log_odds)
        self._trace(f"  - Prediksi Tree ke-{0}   : {[round(val, 2) for val in np.array(sigmoid(F))[:5]]}")
        for i, tree in enumerate(self.trees[:self.best_iter + 1]):
            tree_pred = tree.predict(X, decode=False)
            F += self.learning_rate * tree_pred
            label = f"Prediksi Tree ke-{i + 1}".ljust(20)
            self._trace(f"  - {label} : {[round(val, 2) for val in np.array(sigmoid(F))[:5]]}")
        
        probs = sigmoid(F)
        predprint = (probs >= threshold).astype(int)
        self._trace(f"  - Hasil Prediksi       : {predprint[:5]}")
        if decode and self.target_encoder:
            decoder = {encoded_val: original_label for original_label, encoded_val in self.target_encoder.items()}
            preds = [decoder.get(int(prob >= threshold), int(prob >= threshold)) for prob in probs]
            
        else:
            preds = (probs >= threshold).astype(int)
        return preds 

    def _calculate_obj(self, tree):
        """
        Menghitung nilai objektif untuk pohon yang ditentukan.

        Nilai objektif dihitung dengan menjumlahkan loss pada setiap node daun,
        dengan mempertimbangkan parameter regularisasi dan struktur pohon.

        Args:
            tree (DecisionTreeXGB): Pohon keputusan XGBoost yang telah dilatih.

        Returns:
            float: Nilai objektif dari pohon yang ditentukan.
        """
        leaf_nodes = tree.get_leaf_nodes()
        obj_value = -0.5 * sum(node.G ** 2 / (node.H + self.lambda_reg) for node in leaf_nodes) - self.gamma * len(leaf_nodes)
        return obj_value
