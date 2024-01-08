
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class RegressionTree:
    """
    Implementasi pohon regresi.

    Pohon ini digunakan untuk memprediksi nilai kontinu berdasarkan fitur yang diberikan.
    Pohon dibangun dengan membagi data secara rekursif untuk meminimalkan kesalahan kuadrat rata-rata (MSE).

    Attributes:
        max_depth (int): Kedalaman maksimum pohon.
        auto_encode (bool): Otomatis melakukan encoding fitur kategorikal jika True.
        verbose (bool): Mencetak informasi detail selama proses fitting jika True.
        tree (Node): Akar pohon regresi.
        feature_encoders (dict): Encoder untuk fitur kategorikal.
        target_encoder (dict): Encoder untuk target kategorikal.
        name (str): Nama pohon (opsional).
    """
    def __init__(self, max_depth=None,auto_encode=True,verbose=False):
        """
        Inisialisasi pohon regresi.

        Args:
            max_depth (int, optional): Kedalaman maksimum pohon. Jika None, tidak ada batasan kedalaman.
            auto_encode (bool, optional): Jika True, melakukan encoding fitur kategorikal secara otomatis.
            verbose (bool, optional): Jika True, mencetak informasi detail selama proses fitting.

        Attributes:
            max_depth (int): Kedalaman maksimum pohon.
            tree (Node): Akar pohon regresi.
            feature_encoders (dict): Encoder untuk fitur kategorikal.
            target_encoder (dict): Encoder untuk target kategorikal.
            name (str): Nama pohon (opsional).
            auto_encode (bool): Otomatis melakukan encoding fitur kategorikal jika True.
            verbose (bool): Mencetak informasi detail selama proses fitting jika True.
        """
        self.max_depth = max_depth
        self.tree = None
        self.feature_encoders = {}
        self.target_encoder = None
        self.name=None
        self.auto_encode=auto_encode
        self.verbose = verbose



    class Node:
        """
            Representasi dari sebuah node dalam pohon regresi.

            Setiap node dapat menjadi node internal yang memiliki kriteria split, 
            atau node daun yang memiliki nilai prediksi.

            Args:
                feature_name (str, optional): Nama fitur yang digunakan untuk pemisahan di node ini.
                threshold (float, optional): Nilai threshold untuk pemisahan.
                left (Node, optional): Node anak di sisi kiri (nilai fitur <= threshold).
                right (Node, optional): Node anak di sisi kanan (nilai fitur > threshold).
                value (float, optional): Nilai prediksi untuk node daun.
                left_label (str, optional): Label untuk cabang kiri (untuk visualisasi).
                right_label (str, optional): Label untuk cabang kanan (untuk visualisasi).
                count (int, optional): Jumlah sampel di node ini.

            Attributes:
                feature_name (str): Nama fitur untuk pemisahan.
                threshold (float): Threshold untuk pemisahan.
                left (Node): Anak kiri node.
                right (Node): Anak kanan node.
                value (float): Nilai prediksi node daun.
                left_label (str): Label untuk cabang kiri.
                right_label (str): Label untuk cabang kanan.
                count (int): Jumlah sampel di node.
            """
        def __init__(self, feature_name=None, threshold=None, left=None, right=None, value=None,left_label=None,right_label=None,count=None):
            self.feature_name = feature_name
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.left_label=left_label
            self.right_label=right_label
            self.count=count
            

    def title(self,title):
        """
        Menetapkan judul untuk pohon keputusan (opsional).

        Args:
            title: Judul yang akan ditetapkan.
        """
        self.name=title

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
        Mengenkripsi data fitur dan target.

        Jika fitur atau target adalah kategorikal, fungsi ini akan mengenkripsi
        data tersebut menjadi bentuk numerik menggunakan pemetaan yang telah ditentukan.

        Args:
            X: DataFrame yang berisi fitur.
        """
        if len(self.feature_encoders) == 0:
            for col in X.columns:
                if X[col].dtype == object:
                    self.feature_encoders[col] = {val: idx for idx, val in enumerate(X[col].unique())}
        
        if self.target_encoder is None and y.dtype == object:
            self.target_encoder = {val: idx for idx, val in enumerate(y.unique())}

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

   


    def fit(self, X, y, feature_encoders={}, target_encoder=None, depth=0):
        """
        Melatih pohon regresi menggunakan data fitur dan target.

        Args:
            X (DataFrame): Fitur data.
            y (Series): Target data.
            feature_encoders (dict, optional): Encoder untuk fitur kategorikal. Default kosong.
            target_encoder (dict, optional): Encoder untuk target kategorikal. Default kosong.
            depth (int, optional): Kedalaman awal (digunakan secara internal). Default 0.
        """
        if self.auto_encode:
            self.feature_encoders = feature_encoders
            self.target_encoder = target_encoder
            self._encode_data(X, y)
        
        
        # Menggunakan data yang sudah dienkodikan jika tersedia
        if self.feature_encoders:
            encoded_X = X.apply(lambda col: self._encode_column(col, self.feature_encoders[col.name]) if col.name in self.feature_encoders and col.dtype == object else col)
        else:
            encoded_X = X
        if self.target_encoder:
            encoded_y = self._encode_column(y, self.target_encoder) if self.target_encoder and y.dtype == object else y
        else:
            encoded_y = y

        # Unik untuk y yang dienkodikan
        uniq = np.unique(encoded_y)

        # Membangun pohon
        self.tree = self._build_tree(encoded_X, encoded_y, depth, uniq)


    def get_label(self, X, feature_name, mappings):
        """
        Menghasilkan label untuk cabang berdasarkan feature dan mappings.

        Args:
            X (DataFrame): Fitur data.
            feature_name (str): Nama fitur yang digunakan untuk split.
            mappings (dict): Pemetaan untuk feature kategorikal.

        Returns:
            str: Label untuk cabang.
        """
        feature_mapping = mappings.get(feature_name, {})
        reverse_mapping = {v: k for k, v in feature_mapping.items()}
        X_values = [reverse_mapping.get(i, i) for i in X[feature_name].unique()]
        return ", ".join(X_values)


    def _build_tree(self, X, y, depth, uniq):
        """
        Membangun pohon regresi secara rekursif.

        Args:
            X (DataFrame): Fitur data.
            y (Series): Target data.
            depth (int): Kedalaman saat ini dalam pohon.
            uniq (numpy.array): Nilai unik dalam target.

        Returns:
            Node: Node pohon regresi.
        """
        num_samples = len(y)
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples <= 1:
            leaf_value = self._calculate_leaf_value(y)
            if self.verbose:
                self._trace(f"Mencapai daun pada kedalaman {depth}. Nilai daun: {leaf_value}, Jumlah Sampel: {num_samples}")
            return self.Node(value=leaf_value, count=num_samples)

        best_feature_name, best_threshold = self._get_best_split(X, y, depth)
        X_left, y_left, X_right, y_right = self._split(X, y, best_threshold, best_feature_name)

        if best_feature_name in self.feature_encoders:
            label_left = self.get_label(X_left, best_feature_name, self.feature_encoders)
            label_right = self.get_label(X_right, best_feature_name, self.feature_encoders)
        else:
            label_left = f' <= {round(best_threshold, 2)}'
            label_right = f' > {round(best_threshold, 2)}'

        if self.verbose:
            self._trace(f"Membangun sub-pohon kiri dan kanan pada kedalaman {depth + 1}")

        left = self._build_tree(X_left, y_left, depth + 1, uniq)
        right = self._build_tree(X_right, y_right, depth + 1, uniq)

        return self.Node(best_feature_name, best_threshold, left, right, left_label=label_left, right_label=label_right, count=num_samples)
       


    def _calculate_leaf_value(self, y):
        """
        Menghitung nilai untuk node daun berdasarkan target y.

        Nilai daun dihitung sebagai rata-rata target pada node tersebut.

        Args:
            y (numpy.array): Nilai target di node daun.

        Returns:
            float: Nilai rata-rata target.
        """
        # Menghitung rata-rata target pada daun
        return np.mean(y)


    def _get_best_split(self, X, y, depth):
        """
        Mencari split terbaik berdasarkan MSE.

        Args:
            X (DataFrame): Fitur data.
            y (Series): Target data.
            depth (int): Kedalaman saat ini dalam pohon.

        Returns:
            Tuple[str, float]: Nama fitur terbaik untuk split dan nilai threshold.
        """
        best_feature_name = None
        best_threshold = None
        best_mse = float("inf")
        split_info = []

        for feature_name in X.columns:
            unique_values = np.sort(X[feature_name].unique())
            # Menggunakan metode kuantil jika banyak nilai unik
            if len(unique_values) > 10:
                quantiles = np.linspace(0, 1, 11)  # 10 kuantil (11 titik termasuk 0 dan 1)
                quantile_values = np.quantile(unique_values, quantiles)
                thresholds=quantile_values
                # thresholds = (quantile_values[:-1] + quantile_values[1:]) / 2  # Nilai tengah antara kuantil berurutan
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # Nilai tengah antara nilai unik berurutan

            for threshold in thresholds:
                mse = self._mean_squared_error(y, X, threshold, feature_name)
                split_info.append((feature_name, threshold, mse))

                if mse < best_mse:
                    best_mse = mse
                    best_feature_name = feature_name
                    best_threshold = threshold

        if self.verbose:
            self._trace(f"\nPencarian Split pada Kedalaman {depth}:")
            split_df = pd.DataFrame(split_info, columns=['Feature', 'Threshold', 'MSE'])
            self._trace(split_df)
            self._trace(f"Best Split: Feature {best_feature_name}, Threshold {best_threshold}, MSE {best_mse}")

        return best_feature_name, best_threshold


    def _split(self, X, y, threshold, feature_name):
        """
        Memisahkan data berdasarkan threshold dan fitur yang diberikan.

        Args:
            X (DataFrame): Fitur data.
            y (Series): Target data.
            threshold (float): Nilai threshold untuk pemisahan.
            feature_name (str): Nama fitur yang digunakan untuk pemisahan.

        Returns:
            Tuple[DataFrame, Series, DataFrame, Series]: 
            Dataset yang terpisah (X_left, y_left, X_right, y_right).
        """
        # Membuat masker boolean untuk membagi data
        left_mask = X[feature_name] <= threshold
        right_mask = X[feature_name] > threshold

        # Memisahkan X dan y berdasarkan masker
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return X_left, y_left, X_right, y_right

    def _mean_squared_error(self, y, X, threshold, feature_name):
        """
        Menghitung mean squared error (MSE) dari split yang diusulkan.

        Args:
            y (Series): Target data.
            X (DataFrame): Fitur data.
            threshold (float): Nilai threshold untuk pemisahan.
            feature_name (str): Nama fitur yang digunakan untuk pemisahan.

        Returns:
            float: MSE dari split yang diusulkan.
        """
        left_mask = X[feature_name] <= threshold
        right_mask = X[feature_name] > threshold
        y_left, y_right = y[left_mask], y[right_mask]

        mse_left = np.mean((y_left - np.mean(y_left)) ** 2) if len(y_left) > 0 else 0
        mse_right = np.mean((y_right - np.mean(y_right)) ** 2) if len(y_right) > 0 else 0
        mse = len(y_left) / len(y) * mse_left + len(y_right) / len(y) * mse_right
        return mse


    def predict(self, X):
        """
        Membuat prediksi berdasarkan pohon regresi yang telah dilatih.

        Args:
            X (DataFrame): Fitur data untuk prediksi.

        Returns:
            List[float]: Prediksi untuk setiap sampel dalam X.
        """
        encoded_X = X.apply(lambda col: self._encode_column(col, self.feature_encoders.get(col.name, lambda x: x)))
        predictions = [self._traverse_tree(row, self.tree) for _, row in encoded_X.iterrows()]
        return predictions



    def _traverse_tree(self, x, node):
        """
        Menelusuri pohon regresi untuk membuat prediksi.

        Args:
            x (Series): Satu baris fitur yang akan diprediksi.
            node (Node): Node saat ini dalam pohon.

        Returns:
            float: Prediksi untuk baris x.
        """
        # Kasus basis: Jika node adalah daun
        if node.value is not None:
            return node.value

        # Rekursif menelusuri ke kiri atau kanan berdasarkan nilai fitur dan threshold
        if x[node.feature_name] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

        
    
    def visualize_tree(self):
        """
        Visualisasi struktur pohon regresi yang telah dilatih.

        Menggunakan matplotlib untuk menampilkan visualisasi pohon.
        """
        print("\nVisualisasi Tree")
        def draw_node(ax, node, x, y, dx, dy,scale):
            scale=4
            node_width = 0.2*scale*2  # Lebar kotak untuk node
            node_height = 0.1*scale*2  # Tinggi kotak untuk node
            leaf_width = 0.05*scale*2  # Lebar belah ketupat untuk daun
            leaf_height = 0.05*scale*2  # Tinggi belah ketupat untuk daun
            vertical_gap = 0.2*scale*2  # Jarak vertikal antar level
            text_offset = 0.05*scale  # Jarak vertikal teks dari garis
            if node is None:
                return

            if node.value is not None:  # Leaf node
                diamond = patches.RegularPolygon((x, y), numVertices=4, radius=leaf_width, orientation=np.pi/2,
                                                facecolor='skyblue', edgecolor='black')
                ax.add_patch(diamond)

                labels = f"{node.value:.2f}"
                plt.text(x, y,f"\n{labels} \n[{node.count}]\n" , ha='center', va='center', fontsize=7.5)
        
                diamond = patches.RegularPolygon((x, y), numVertices=4, radius=leaf_width, orientation=np.pi/2,
                                                facecolor='skyblue', edgecolor='black')
               
            else:  # Internal node
                rect = patches.Rectangle((x - node_width/2, y - node_height/2), node_width, node_height,facecolor='lightgreen', edgecolor='black')
                ax.add_patch(rect)
                plt.text(x, y, f"{node.feature_name} \n[{node.count}]", ha='center', va='center', fontsize=10)

                # Rekursif untuk anak-anak node
                if node.left is not None:
                    left_x = x - dx/2 
                    left_y = y - vertical_gap + leaf_height/2
                    plt.plot([x, left_x], [y - node_height/2, left_y + node_height/2], '-', color='grey') 
                    plt.text((x + left_x) / 2 - (0.3*dx), y - node_height/2 - text_offset, f'{node.left_label}\n',ha='center', va='top', fontsize=8, color='red')

                    draw_node(ax,node.left, left_x, left_y, dx/2, dy/2,scale)

                if node.right is not None:
                    right_x = x + dx/2 
                    right_y = y - vertical_gap + leaf_height/2
                    plt.plot([x, right_x], [y - node_height/2, right_y+node_height/2], '-', color='grey')
                    plt.text((x + right_x ) / 2 + (0.3*dx), y - node_height/2 - text_offset, f'{node.right_label}',ha='center', va='top', fontsize=8, color='blue')
                    draw_node(ax, node.right, right_x, right_y, dx/2, dy/2,scale)
      

        # Hitung kedalaman maksimum pohon
        max_depth = self.calculate_max_depth(self.tree)
        max_depth=max_depth+1
        fig_width = max(6, max_depth * 3)  # Contoh: lebar 3 unit per level kedalaman
        fig_height = max(4, max_depth * 2)  # Contoh: tinggi 2 unit per level kedalaman

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        initial_x = fig_width / 2
        initial_y = fig_height - 1  # Biarkan ruang di bagian atas kanvas
        dx = 0.2*(max_depth+1)**2  # Lebar kotak untuk node (dapat diatur sesuai kebutuhan)
        dy = 0.2  # Tinggi kotak untuk node (dapat diatur sesuai kebutuhan)

        draw_node(ax, self.tree, initial_x, initial_y, dx, dy,scale=max_depth+1)

        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.axis('off')
        if self.name:
            plt.title(self.name)
        else:
            plt.title("Regression Tree")
        plt.show()

    
    def calculate_max_depth(self, node):
        """
        Menghitung kedalaman maksimum dari pohon keputusan.

        Args:
            node (Node, optional): Node saat ini dalam pohon. Default root node.

        Returns:
            int: Kedalaman maksimum dari pohon.
        """
        if node is None or node.value is not None:
            return 0
        return 1 + max(self.calculate_max_depth(node.left), self.calculate_max_depth(node.right))
