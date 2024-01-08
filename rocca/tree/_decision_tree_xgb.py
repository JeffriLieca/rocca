
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DecisionTreeXGB:
    """
    Implementasi Decision Tree untuk XGBoost.

    Decision Tree ini dirancang untuk digunakan dalam ensemble model XGBoost,
    dengan fokus pada optimasi berbasis gradien.

    Attributes:
        max_depth (int): Kedalaman maksimum pohon.
        lambda_reg (float): Parameter regularisasi lambda.
        gamma (float): Parameter minimum loss reduction yang diperlukan untuk membuat split.
        min_child_weight (int): Minimum jumlah hessian (second order gradient) yang diperlukan dalam sebuah node.
        tree (Node): Root node dari pohon keputusan.
        feature_encoders (dict): Encoder untuk fitur kategorikal.
        target_encoder (dict): Encoder untuk target kategorikal.
        name (str): Nama pohon (opsional).
        auto_encode (bool): Jika True, melakukan encoding fitur kategorikal secara otomatis.
        verbose (bool): Jika True, mencetak informasi detail selama proses fitting.

    Methods:
        fit(X, y, grad, hess): Melatih pohon keputusan menggunakan data, gradien, dan hessian.
        predict(X): Membuat prediksi berdasarkan pohon keputusan yang telah dilatih.
        visualize_tree(): Visualisasi pohon keputusan yang telah dilatih.
    """
    def __init__(self, max_depth=None, lambda_reg=1.0, gamma=0.0, min_child_weight=0, auto_encode=True, verbose=False,feature_encoders={}, target_encoder=None):
        """
        Inisialisasi DecisionTreeXGB.

        Args:
            max_depth (int, optional): Kedalaman maksimum pohon. Default None (tidak terbatas).
            lambda_reg (float, optional): Parameter regularisasi lambda. Default 1.0.
            gamma (float, optional): Minimum loss reduction yang diperlukan untuk membuat split. Default 0.0.
            min_child_weight (int, optional): Minimum jumlah hessian yang diperlukan dalam node. Default 0.
            auto_encode (bool, optional): Otomatis melakukan encoding fitur kategorikal. Default True.
            verbose (bool, optional): Mencetak informasi detail selama proses fitting. Default False.
            feature_encoders (dict, optional): Encoder untuk fitur kategorikal. Default kosong.
            target_encoder (dict, optional): Encoder untuk target kategorikal. Default kosong.
        """
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.tree = None
        self.feature_encoders = feature_encoders
        self.target_encoder = target_encoder
        self.name = None
        self.auto_encode = auto_encode
        self.verbose = verbose

    class Node:
        """
        Representasi dari sebuah node dalam pohon keputusan XGBoost.

        Setiap node dapat menjadi node internal yang memiliki kriteria split, 
        atau node daun yang memiliki nilai prediksi.

        Attributes:
            feature_name (str, optional): Nama fitur yang digunakan untuk pemisahan di node ini.
            threshold (float, optional): Nilai threshold untuk pemisahan.
            left (Node, optional): Node anak di sisi kiri (nilai fitur <= threshold).
            right (Node, optional): Node anak di sisi kanan (nilai fitur > threshold).
            value (float, optional): Nilai prediksi untuk node daun.
            left_label (str, optional): Label untuk cabang kiri (untuk visualisasi).
            right_label (str, optional): Label untuk cabang kanan (untuk visualisasi).
            G (float, optional): Total gradien di node ini.
            H (float, optional): Total hessian di node ini.
            count (str or int): Jumlah sampel atau representasi statistik di node ini.
        """
        def __init__(self, feature_name=None, threshold=None, left=None, right=None, value=None, left_label=None, right_label=None, G=None,H=None, count=None):
            self.feature_name = feature_name
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.left_label = left_label
            self.right_label = right_label
            self.count = count
            self.G = G  
            self.H = H  

    def title(self, title):
        """
        Menetapkan judul untuk pohon keputusan (opsional).

        Args:
            title: Judul yang akan ditetapkan.
        """
        self.name = title

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

    def fit(self, X, y, grad, hess, feature_encoders={}, target_encoder=None, depth=0):
        """
        Melatih Decision Tree menggunakan gradien dan hessian dari loss function.

        Berbeda dengan pohon keputusan tradisional, pohon ini menggunakan gradien dan hessian 
        dari loss function untuk menentukan split terbaik, yang sangat penting dalam konteks 
        boosting.

        Args:
            X (DataFrame): Fitur training.
            y (Series): Target training.
            grad (numpy.array): Gradien dari loss function.
            hess (numpy.array): Hessian dari loss function.
            feature_encoders (dict, optional): Encoder untuk fitur kategorikal. Default kosong.
            target_encoder (dict, optional): Encoder untuk target kategorikal. Default kosong.
            depth (int, optional): Kedalaman awal (internal use). Default 0.
        """
        self.actual_max_depth = 0  # Inisialisasi di awal fit
        if self.auto_encode:
            self.feature_encoders = feature_encoders
            self.target_encoder = target_encoder
            self._encode_data(X, y)

        encoded_X = X.apply(lambda col: self._encode_column(col, self.feature_encoders[col.name]) if col.name in self.feature_encoders and col.dtype == object else col)
        encoded_y = self._encode_column(y, self.target_encoder) if self.target_encoder and y.dtype == object else y

        uniq = np.unique(encoded_y)
        if len(uniq)==2:
            uniq.sort()
        self.tree = self._build_tree(encoded_X, encoded_y, grad, hess, 0, uniq)

    
    def _calculate_leaf_value(self, grad, hess):
        """
        Menghitung nilai untuk node daun berdasarkan gradien dan hessian.

        Nilai daun dihitung untuk memaksimalkan penurunan loss dalam konteks XGBoost.

        Args:
            grad (numpy.array): Gradien dari loss function pada node.
            hess (numpy.array): Hessian dari loss function pada node.

        Returns:
            Tuple[float, float, float]: Nilai daun, total gradien, dan total hessian.
        """
        G = grad.sum()
        H = hess.sum()
        leaf_value = -G / (H + self.lambda_reg)
        return leaf_value, G, H

    
    def get_leaf_nodes(self, node=None):
        """
        Mengambil semua node daun dari pohon keputusan.

        Metode ini digunakan untuk mendapatkan node-node daun dalam pohon, yang berguna
        untuk perhitungan objektif dan visualisasi.

        Args:
            node (Node, optional): Node saat ini. Jika None, akan mulai dari root.

        Returns:
            List[Node]: Daftar semua node daun dalam pohon.
        """
        if node is None:
            node = self.tree

        # Jika node adalah daun, kembalikan dalam list
        if node.value is not None:
            return [node]

        # Jika bukan daun, lanjutkan rekursi ke anak-anak
        leaf_nodes = []
        if node.left is not None:  # Jika ada anak kiri, rekursi ke anak kiri
            leaf_nodes.extend(self.get_leaf_nodes(node.left))

        if node.right is not None:  # Jika ada anak kanan, rekursi ke anak kanan
            leaf_nodes.extend(self.get_leaf_nodes(node.right))

        return leaf_nodes

    def _calculate_gain(self, X, y, grad, hess, threshold, feature_name):
        """
        Menghitung gain dari split berdasarkan feature dan threshold tertentu.

        Gain dihitung dengan mempertimbangkan gradien dan hessian dari loss function,
        yang menunjukkan penurunan loss akibat split tersebut.

        Args:
            X (DataFrame): Fitur training.
            y (Series): Target training.
            grad (numpy.array): Gradien dari loss function.
            hess (numpy.array): Hessian dari loss function.
            threshold (float): Nilai threshold untuk split.
            feature_name (str): Nama fitur yang digunakan untuk split.

        Returns:
            float: Gain dari split yang diusulkan.
        """
        left_indices = X[feature_name] <= threshold
        G_L, H_L = grad[left_indices].sum(), hess[left_indices].sum()
        G_R, H_R = grad[~left_indices].sum(), hess[~left_indices].sum()
        gain = 0.5 * ((G_L ** 2 / (H_L + self.lambda_reg)) + (G_R ** 2 / (H_R + self.lambda_reg)) - ((G_L + G_R) ** 2 / (H_L + H_R + self.lambda_reg))) - self.gamma
        if gain < self.gamma and self.gamma != 0:
            return -np.inf  # Split tidak layak karena tidak memenuhi gamma
        return max(gain, 0)  # Pastikan gain tidak negatif

    def _build_tree(self, X, y, grad, hess, depth, uniq):
        """
        Membangun pohon keputusan secara rekursif.

        Metode ini memilih split terbaik pada setiap tingkat pohon berdasarkan gain informasi,
        yang dihitung dari gradien dan hessian.

        Args:
            X (DataFrame): Fitur data.
            y (Series): Target data.
            grad (numpy.array): Gradien dari loss function.
            hess (numpy.array): Hessian dari loss function.
            depth (int): Kedalaman saat ini dalam pohon.
            uniq (numpy.array): Nilai unik dalam target.

        Returns:
            Node: Node pohon keputusan.
        """
        self._trace("")
        num_samples = len(y)
        counts = [np.sum(y == unique_value) for unique_value in uniq]
        if len(counts)==2:
            output = "[" + str(counts[1]) + "+," + str(counts[0]) + "-]"
        else :
            output=[num_samples]

        
        if len(np.unique(y)) == 1 :
            leaf_value, G, H = self._calculate_leaf_value(grad, hess)
            self._trace(f"Mencapai leaf dengan nilai leaf: {round(leaf_value, 2)}, count: {output}")
            return self.Node(value=leaf_value, G=G, H=H, count=output)

        # Cek kondisi berhenti
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_value, G, H = self._calculate_leaf_value(grad, hess)
            self._trace(f"Kedalaman maksimum tercapai ({depth}). Menghentikan dengan nilai leaf: {round(leaf_value, 2)}, count: {output}")
            return self.Node(value=leaf_value, G=G, H=H, count=output)
        
        # Periksa kondisi min_child_weight
        if hess.sum() < self.min_child_weight:
            leaf_value, G, H = self._calculate_leaf_value(grad, hess)
            self._trace(f"Berhenti karena min_child_weight di kedalaman {depth}. Menghentikan dengan nilai leaf: {round(leaf_value, 2)}, count: {output}")
            return self.Node(value=leaf_value, G=G, H=H, count=output)
        
        


        best_feature_name, best_threshold,best_gain = self._get_best_split(X, grad, hess, depth)
        # Jika gain terbaik adalah -np.inf, berarti split tidak layak
        if best_gain == -np.inf:
            leaf_value, G, H = self._calculate_leaf_value(grad, hess)
            self._trace(f"Split tidak layak (gain < gamma) di kedalaman {depth}. Membuat leaf node dengan nilai leaf: {round(leaf_value, 2)}, count: {output}")
            return self.Node(value=leaf_value, G=G, H=H, count=output)
        self._trace(f"Best Split: Feature {best_feature_name}, Threshold {best_threshold}, Gain {round(best_gain,2)}")

        X_left, y_left, X_right, y_right, grad_left, hess_left, grad_right, hess_right = self._split(X, y, grad, hess, best_threshold, best_feature_name)

        label_left, label_right = self._get_split_labels(X, best_feature_name, best_threshold)
        left = self._build_tree(X_left, y_left, grad_left, hess_left, depth + 1, uniq)
        right = self._build_tree(X_right, y_right, grad_right, hess_right, depth + 1, uniq)
        self.actual_max_depth = max(self.actual_max_depth, depth)
        return self.Node(best_feature_name, best_threshold, left, right, left_label=label_left, right_label=label_right, count=output)

    def _get_split_labels(self, X, feature_name, threshold):
        """
        Menghasilkan label untuk split berdasarkan feature dan threshold.

        Label ini berguna untuk visualisasi dan interpretasi split dalam pohon keputusan.

        Args:
            X (DataFrame): Fitur data.
            feature_name (str): Nama fitur yang digunakan untuk split.
            threshold (float): Nilai threshold untuk split.

        Returns:
            Tuple[str, str]: Label untuk cabang kiri dan kanan split.
        """
        if feature_name in self.feature_encoders:
            feature_mapping = self.feature_encoders[feature_name]
            reverse_mapping = {idx: val for val, idx in feature_mapping.items()}
            left_labels = ", ".join([reverse_mapping.get(i, i) for i in X[feature_name].unique() if i <= threshold])
            right_labels = ", ".join([reverse_mapping.get(i, i) for i in X[feature_name].unique() if i > threshold])
        else:
            left_labels = f'<= {threshold}'
            right_labels = f'> {threshold}'
        return left_labels, right_labels


    def _get_best_split(self, X, grad, hess, depth):
        """
        Mencari split terbaik berdasarkan gradien dan hessian.

        Metode ini mencari melalui setiap fitur dan threshold yang mungkin
        untuk menemukan kombinasi yang memberikan gain informasi terbesar.

        Args:
            X (DataFrame): Fitur data.
            grad (numpy.array): Gradien dari loss function.
            hess (numpy.array): Hessian dari loss function.
            depth (int): Kedalaman saat ini dalam pohon.

        Returns:
            Tuple[str, float, float]: Terbaik fitur untuk split, threshold, dan gain.
        """
        best_gain = -float("inf")
        best_feature_name = None
        best_threshold = None
        split_info = []
        for feature_name in X.columns:
            unique_values = np.sort(X[feature_name].unique())
             # Menggunakan metode kuantil jika banyak nilai unik
            if len(unique_values) > 10:
                quantiles = np.linspace(0, 1, 11)  # 10 kuantil
                quantile_values = np.quantile(unique_values, quantiles)
                thresholds = (quantile_values[:-1] + quantile_values[1:]) / 2  # Rata-rata antara kuantil berurutan
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # Nilai tengah antara nilai unik berurutan

            for threshold in thresholds:
                gain = self._calculate_gain(X, X[feature_name], grad, hess, threshold, feature_name)
                split_info.append({
                    'Feature': feature_name,
                    'Threshold': threshold,
                    'Gain': round(gain, 2)
                })

                if gain > best_gain:
                    best_gain = gain
                    best_feature_name = feature_name
                    best_threshold = threshold
        
        if self.verbose:
                self._trace(f"Informasi Split pada Kedalaman {depth}:")
                df_split_info = pd.DataFrame(split_info)
                self._trace(df_split_info)
                
        return best_feature_name, best_threshold, best_gain

    def _split(self, X, y, grad, hess, threshold, feature_name):
        """
        Memisahkan data berdasarkan threshold dan fitur yang diberikan.

        Args:
            X (DataFrame): Fitur data.
            y (Series): Target data.
            grad (numpy.array): Gradien dari loss function.
            hess (numpy.array): Hessian dari loss function.
            threshold (float): Nilai threshold untuk pemisahan.
            feature_name (str): Nama fitur yang digunakan untuk pemisahan.

        Returns:
            Tuple[DataFrame, Series, DataFrame, Series, numpy.array, numpy.array, numpy.array, numpy.array]: 
            Dataset yang terpisah (X_left, y_left, X_right, y_right) dan gradien dan hessian yang sesuai.
        """
        left_mask = X[feature_name] <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        grad_left, hess_left = grad[left_mask], hess[left_mask]
        grad_right, hess_right = grad[right_mask], hess[right_mask]

        return X_left, y_left, X_right, y_right, grad_left, hess_left, grad_right, hess_right

    

    def predict(self, X, decode=True):
        """
        Membuat prediksi nilai berdasarkan pohon keputusan yang telah dilatih.

        Berbeda dengan pohon keputusan tradisional yang menghasilkan kelas, pohon ini 
        menghasilkan nilai prediksi yang kemudian akan digunakan dalam ensemble XGBoost.

        Args:
            X (DataFrame): Fitur untuk prediksi.
            decode (bool, optional): Mendekode prediksi ke label asli jika True. Default True.

        Returns:
            numpy.array: Nilai prediksi untuk setiap sampel dalam X.
        """
        encoded_X = X.apply(lambda col: self._encode_column(col, self.feature_encoders[col.name]) if col.name in self.feature_encoders and col.dtype == object else col)
        predictions = np.array([self._traverse_tree(row, self.tree) for _, row in encoded_X.iterrows()])
        if decode:
            decoded_predictions = np.array([self._decode_prediction(pred, self.target_encoder) for pred in predictions])
        else:
            decoded_predictions = predictions
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

    def _traverse_tree(self, x, node):
        """
        Menelusuri pohon keputusan untuk membuat prediksi.

        Args:
            x (Series): Satu baris fitur yang akan diprediksi.
            node (Node): Node saat ini dalam pohon.

        Returns:
            float: Prediksi untuk baris x.
        """
        if node.value is not None:
            return node.value

        if x[node.feature_name] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def visualize_tree(self):
        """
        Visualisasi struktur pohon keputusan yang telah dilatih.

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

                
                labels=round(node.value, 2)
                plt.text(x, y,f"{labels} \n{node.count}\n" , ha='center', va='center', fontsize=7.5)
                diamond = patches.RegularPolygon((x, y), numVertices=4, radius=leaf_width, orientation=np.pi/2,
                                                facecolor='skyblue', edgecolor='black')
               
            else:  # Internal node
                rect = patches.Rectangle((x - node_width/2, y - node_height/2), node_width, node_height,facecolor='lightgreen', edgecolor='black')
                ax.add_patch(rect)
                plt.text(x, y, f"{node.feature_name} \n{node.count}", ha='center', va='center', fontsize=10)

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
        
        fig_width = max(6, max_depth * 3)  
        fig_height = max(4, max_depth * 2)  

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        initial_x = fig_width / 2
        initial_y = fig_height - 1 
        dx = 0.2*(max_depth+1)**2 
        dy = 0.2  

        draw_node(ax, self.tree, initial_x, initial_y, dx, dy,scale=max_depth+1)

        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.axis('off')
        if self.name:
            plt.title(self.name)
        else:
            plt.title("Decision Tree")
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

