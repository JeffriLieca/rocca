
import numpy as np


class BaggingClassifier:
    """
    Implementasi algoritma Bagging untuk klasifikasi.

    Bagging, singkatan dari Bootstrap Aggregating, adalah metode ensemble learning yang
    bertujuan untuk meningkatkan stabilitas dan akurasi dari algoritma machine learning.
    Metode ini menggabungkan beberapa model yang sama untuk mendapatkan prediksi akhir
    yang lebih kuat dan stabil.

    Attributes:
        base_classifier: Classifier dasar yang akan digunakan dalam ensemble.
        n_estimators: Jumlah classifier yang akan digunakan dalam ensemble.
        classifiers: Daftar classifier yang dibuat selama proses fitting.
        random_state: Seed untuk generator bilangan acak.
        feature_encoders: Encoder untuk fitur kategorikal.
        target_encoder: Encoder untuk target.
        max_depth: Kedalaman maksimum untuk pohon keputusan dalam base classifier.
        verbose: Jika True, mencetak informasi detail selama proses fitting dan prediksi.
    """
    def __init__(self, base_classifier, n_estimators,random_state=None, feature_encoders={}, target_encoder=None,max_depth=None, verbose=False):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.classifiers = []
        self.random_state = random_state
        self.feature_encoders = feature_encoders
        self.target_encoder = target_encoder
        self.max_depth=max_depth
        self.verbose=verbose

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

    def _initialize_classifier(self):
        """
        Menginisialisasi classifier dasar yang akan digunakan dalam ensemble.

        Metode ini memeriksa apakah `base_classifier` adalah tipe kelas atau instance.
        Jika merupakan tipe kelas, metode ini akan membuat instance baru dari kelas tersebut.
        Jika merupakan instance, metode ini akan membuat salinan independen dari instance tersebut.

        Returns:
            Instance dari classifier dasar yang digunakan.
        """
        if isinstance(self.base_classifier, type):
            return self.base_classifier(max_depth=self.max_depth, verbose=self.verbose)
        else:
            return type(self.base_classifier)(max_depth=self.max_depth, verbose=self.verbose)
    def fit(self, X, y):
        """
        Fungsi untuk melatih Bagging Classifier.

        Metode ini akan membuat beberapa classifier berdasarkan base_classifier yang
        diberikan. Setiap classifier akan dilatih pada sampel data yang di-bootstrap.

        Args:
            X: Fitur training.
            y: Target training.
        """
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        self._encode_data(X, y)

        encoded_X = X.apply(lambda col: self._encode_column(col, self.feature_encoders[col.name]) if col.name in self.feature_encoders and col.dtype == object else col)
        encoded_y = self._encode_column(y, self.target_encoder) if self.target_encoder and y.dtype == object else y

        np.random.seed(self.random_state)

        for i in range(self.n_estimators):
            self._trace(f"--- Iterasi ke-{i+1}")

            indices = np.random.choice(len(encoded_X), len(encoded_X), replace=True)
            X_sampled, y_sampled = encoded_X.iloc[indices], encoded_y.iloc[indices]

            if self.verbose:
                self._trace("Overview data hasil bootstrap sampling:")
                self._trace(X_sampled.head())
                self._trace(y_sampled.head())

            classifier = self._initialize_classifier()
            classifier.fit(X_sampled, y_sampled,feature_encoders=self.feature_encoders,target_encoder=self.target_encoder)

            # Mencoba visualisasi pohon, jika metode tersedia
            if self.verbose:
                try:
                    classifier.visualize_tree()
                except AttributeError:
                    self._trace("Metode visualize_tree tidak tersedia untuk classifier ini.")
                except Exception as e:
                    self._trace(f"Terjadi error saat visualisasi: {e}")

            self.classifiers.append(classifier)



  
    def predict(self, X):
        """
        Fungsi untuk membuat prediksi dengan Bagging Classifier.

        Metode ini mengagregasi prediksi dari setiap classifier dalam ensemble dan
        mengembalikan prediksi mayoritas.

        Args:
            X: Fitur yang akan diprediksi.
        
        Returns:
            Prediksi untuk setiap sampel dalam X.
        """
        X = X.reset_index(drop=True)
        encoded_X = X.apply(lambda col: self._encode_column(col, self.feature_encoders.get(col.name, lambda x: x)))

        predictions = np.array([classifier.predict(encoded_X, False) for classifier in self.classifiers])

        if self.verbose:
            self._trace("Prediksi dari setiap classifier:")
            for i, pred in enumerate(predictions):
                self._trace(f"Classifier {i+1}: {pred}")

        majority_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=np.max(x)+1).argmax(), axis=0, arr=predictions)

        if self.verbose:
            self._trace("Voting mayoritas untuk setiap sampel:")
            self._trace(majority_votes)
            self._trace("")

        decoded_predictions = [self._decode_prediction(pred, self.target_encoder) for pred in majority_votes]
        return decoded_predictions


    def _decode_prediction(self, prediction, encoder):
        """
        Mendekode prediksi dari representasi numerik ke label asli.

        Fungsi ini digunakan untuk mengubah prediksi yang telah dienkripsi kembali 
        ke bentuk label aslinya berdasarkan encoder yang diberikan.

        Args:
            prediction: Prediksi yang dienkripsi.
            encoder: Kamus yang memetakan nilai asli ke representasi numerik.

        Returns:
            Prediksi dalam bentuk label asli.
        """
        if encoder:
            decoder = {encoded_val: original_label for original_label, encoded_val in encoder.items()}
            return decoder.get(prediction, prediction)
        return prediction

