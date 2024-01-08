import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class DecisionTree:
    """
    Implementasi pohon keputusan untuk klasifikasi.

    Pohon keputusan adalah model prediktif yang menggunakan set aturan keputusan
    untuk memprediksi nilai target berdasarkan fitur.

    Attributes:
        max_depth: Kedalaman maksimum pohon.
        tree: Struktur data yang menyimpan pohon keputusan.
        feature_encoders: Encoder untuk fitur kategorikal.
        target_encoder: Encoder untuk target.
        name: Nama opsional untuk pohon keputusan.
        auto_encode: Jika True, otomatis melakukan encoding pada fitur kategorikal.
        verbose: Jika True, mencetak informasi detail selama proses fitting.
    """
    def __init__(self, max_depth=None,auto_encode=True,verbose=False):
        self.max_depth = max_depth
        self.tree = None
        self.feature_encoders = {}
        self.target_encoder = None
        self.name=None
        self.auto_encode=auto_encode
        self.verbose = verbose



    class Node:
        """
        Representasi dari sebuah node dalam pohon keputusan.

        Setiap node mewakili titik keputusan atau daun dalam pohon keputusan, dengan
        atribut yang menunjukkan fitur dan threshold untuk pemisahan, serta nilai prediksi
        untuk daun.

        Attributes:
            feature_name: Nama fitur yang digunakan untuk pemisahan di node.
            threshold: Nilai threshold untuk pemisahan.
            left: Node anak di sisi kiri (nilai fitur <= threshold).
            right: Node anak di sisi kanan (nilai fitur > threshold).
            value: Nilai prediksi untuk node daun.
            left_label: Label untuk cabang kiri (opsional, untuk visualisasi).
            right_label: Label untuk cabang kanan (opsional, untuk visualisasi).
            count: Jumlah sampel di node (opsional, untuk visualisasi).
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
            y: Series yang berisi target.
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
        Fungsi untuk melatih Decision Tree.

        Metode ini membangun pohon keputusan berdasarkan data training (X dan y).
        Pohon dibangun secara rekursif dengan memilih fitur dan threshold yang
        memberikan gain informasi terbaik pada setiap langkah.

        Args:
            X: Fitur training.
            y: Target training.
        """
        if self.auto_encode:
            self.feature_encoders = feature_encoders
            self.target_encoder = target_encoder
            self._encode_data(X, y)
        
        if self.feature_encoders:
            encoded_X = X.apply(lambda col: self._encode_column(col, self.feature_encoders[col.name]) if col.name in self.feature_encoders and col.dtype == object else col)
        else:
            encoded_X = X
        if self.target_encoder:
            encoded_y = self._encode_column(y, self.target_encoder) if self.target_encoder and y.dtype == object else y
        else:
            encoded_y = y

        uniq = np.unique(encoded_y)
        self.tree = self._build_tree(encoded_X, encoded_y, depth, uniq)


    def get_label(self, X, feature_name, mappings):
        """
        Mendapatkan label yang dapat dibaca untuk fitur kategori pada node.

        Fungsi ini digunakan untuk mendapatkan representasi teks dari nilai kategori
        pada fitur yang digunakan untuk memisahkan pada node.

        Args:
            X: DataFrame yang berisi fitur.
            feature_name: Nama fitur yang sedang diproses.
            mappings: Kamus pemetaan nilai kategori ke representasi numerik.

        Returns:
            String yang mewakili label kategori pada fitur.
        """
        feature_mapping = mappings.get(feature_name, {})
        reverse_mapping = {v: k for k, v in feature_mapping.items()}
        X_values = [reverse_mapping.get(i, i) for i in X[feature_name].unique()]
        return ", ".join(X_values)


    def _build_tree(self, X, y, depth,uniq):
        """
        Membangun pohon keputusan rekursif.

        Args:
            X: Fitur training.
            y: Target training.
            depth: Kedalaman saat ini dalam pohon.
            uniq: Nilai unik dalam target.

        Returns:
            Node: Node pohon keputusan.
        """
        counts = [np.sum(y == unique_value) for unique_value in uniq]
        if len(counts)==2:
            output = "[" + str(counts[1]) + "+," + str(counts[0]) + "-]"
        else :
            output=counts

        num_samples, num_features = X.shape
        if (self.max_depth is not None and depth >= self.max_depth) or len(set(y)) == 1:
            leaf_value = self._calculate_leaf_value(y)
            if self.verbose:
                self._trace(f"Mencapai daun pada kedalaman {depth}. Nilai daun: {leaf_value}")
            return self.Node(value=leaf_value, count=output)

        best_feature_name, best_threshold = self._get_best_split(X, y, depth)
        X_left, y_left, X_right, y_right = self._split(X, y, best_threshold, best_feature_name)
        
        

        if best_feature_name in self.feature_encoders:
            label_left = self.get_label(X_left, best_feature_name, self.feature_encoders)
            label_right = self.get_label(X_right, best_feature_name, self.feature_encoders)          
        else :
            label_left = f' <= {round(best_threshold, 2)}'
            label_right = f' > {round(best_threshold, 2)}'

        if self.verbose:
            self._trace(f"Membangun sub-pohon kiri dan kanan pada kedalaman {depth + 1}")

        # Recursively build the left and right subtrees
        left = self._build_tree(X_left, y_left, depth + 1,uniq)
        right = self._build_tree(X_right, y_right, depth + 1,uniq)

        return self.Node(best_feature_name, best_threshold, left, right,left_label=label_left,right_label=label_right,count=output)
    
       


    def _calculate_leaf_value(self, y):
        """
        Menghitung nilai untuk node daun berdasarkan target (y).

        Fungsi ini mengembalikan nilai yang paling sering muncul dalam target (modus),
        yang akan digunakan sebagai prediksi pada node daun.

        Args:
            y: Kolom target yang berkaitan dengan node daun.

        Returns:
            Nilai yang paling sering muncul dalam y.
        """
        return y.mode()[0]

    

    def _get_best_split(self, X, y, depth):
        """
        Menemukan split terbaik berdasarkan gain informasi untuk fitur pada kedalaman tertentu.

        Fungsi ini mencari melalui setiap fitur dan nilai threshold yang mungkin untuk
        menemukan kombinasi yang memberikan gain informasi terbesar.

        Args:
            X: Fitur training.
            y: Target training.
            depth: Kedalaman saat ini dalam pohon.

        Returns:
            Tuple berisi nama fitur dan threshold untuk split terbaik.
        """
        best_feature_name = None
        best_threshold = None
        best_gain = -float("inf")
        split_info = []

        for feature_name in X.columns:
            unique_values = np.sort(X[feature_name].unique())
            # Menggunakan metode kuantil jika banyak nilai unik
            if len(unique_values) > 10:
                quantiles = np.linspace(0, 1, 11)  # 10 kuantil
                quantile_values = np.quantile(unique_values, quantiles)
                # thresholds = (quantile_values[:-1] + quantile_values[1:]) / 2  
                threshold=quantile_values
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2  

            for threshold in thresholds:
                gain = self._information_gain(y, X, threshold, feature_name)
                split_info.append((feature_name, threshold, gain))
                

                if gain > best_gain:
                    best_gain = gain
                    best_feature_name = feature_name
                    best_threshold = threshold

        if self.verbose:
            self._trace(f"\nPencarian Split pada Kedalaman {depth}:")
            split_df = pd.DataFrame(split_info, columns=['Feature', 'Threshold', 'Gain'])
            self._trace(split_df)
            if best_feature_name in self.feature_encoders:
                self._trace(f"Mapping untuk '{best_feature_name}': {self.feature_encoders[best_feature_name]}")
            self._trace(f"Best Split: Feature {best_feature_name}, Threshold {best_threshold}, Gain {best_gain}")

        return best_feature_name, best_threshold





    def _split(self, X, y, threshold, feature_name):
        """
        Memisahkan data berdasarkan threshold dan fitur yang diberikan.

        Fungsi ini membagi dataset menjadi dua bagian berdasarkan apakah nilai fitur
        lebih kecil atau lebih besar dari threshold.

        Args:
            X: Fitur training.
            y: Target training.
            threshold: Nilai threshold untuk memisahkan data.
            feature_name: Nama fitur yang digunakan untuk pemisahan.

        Returns:
            Empat set data: X_left, y_left, X_right, y_right.
        """

        left_mask = X[feature_name] <= threshold
        right_mask = X[feature_name] > threshold

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return X_left, y_left, X_right, y_right

    def _information_gain(self, y, X, threshold, feature_name):
        """
        Menghitung gain informasi yang dihasilkan dari pemisahan berdasarkan fitur dan threshold.

        Gain informasi diukur sebagai penurunan entropi setelah data dibagi menggunakan
        fitur dan threshold yang ditentukan.

        Args:
            y: Target training.
            X: Fitur training.
            threshold: Nilai threshold untuk pemisahan.
            feature_name: Nama fitur yang digunakan untuk pemisahan.

        Returns:
            Gain informasi dari pemisahan.
        """
        left_mask = X[feature_name] <= threshold
        right_mask = X[feature_name] > threshold
        y_left, y_right = y[left_mask], y[right_mask]

        parent_entropy = self._entropy(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        e_left, e_right = self._entropy(y_left), self._entropy(y_right)
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        ig = parent_entropy - child_entropy
        return ig

    def _entropy(self, y):
        """
        Menghitung entropi dari target (y).

        Entropi adalah ukuran ketidakpastian atau keacakan dalam kumpulan data.
        Fungsi ini menghitung entropi berdasarkan distribusi kelas dalam target.

        Args:
            y: Target training.

        Returns:
            Nilai entropi dari y.
        """
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum(proportions * np.log2(proportions, where=proportions > 0))

        return entropy



    def predict(self, X, decode=True):
        """
        Fungsi untuk membuat prediksi dengan Decision Tree.

        Metode ini menelusuri pohon dari akar hingga daun berdasarkan fitur dan
        nilai threshold pada setiap node untuk menghasilkan prediksi.

        Args:
            X: Fitur yang akan diprediksi.
            decode: Jika True, decode prediksi ke label asli.
        
        Returns:
            Prediksi untuk setiap sampel dalam X.
        """
        
        encoded_X = X.apply(lambda col: self._encode_column(col, self.feature_encoders[col.name]) if col.name in self.feature_encoders and col.dtype == object else col)
        
        predictions = [self._traverse_tree(row, self.tree) for _, row in encoded_X.iterrows()]

        if decode:
            decoded_predictions = [self._decode_prediction(pred, self.target_encoder) for pred in predictions]
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

        Fungsi ini menelusuri pohon keputusan dari akar hingga daun berdasarkan
        nilai fitur pada setiap node untuk menghasilkan prediksi akhir.

        Args:
            x: Satu baris fitur yang akan diprediksi.
            node: Node saat ini dalam pohon.

        Returns:
            Prediksi untuk baris x.
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
        Visualisasi struktur pohon keputusan.

        Metode ini menggambarkan pohon keputusan secara visual menggunakan matplotlib.
        Setiap node dan keputusan ditunjukkan dalam bentuk grafis, memberikan 
        pemahaman intuitif tentang bagaimana pohon keputusan melakukan prediksi.

        Output:
            Visualisasi pohon keputusan dalam bentuk grafis.
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

                if self.target_encoder:
                    reverse_mapping = {v: k for k, v in self.target_encoder.items()}
                    labels = reverse_mapping[node.value]
                else:
                    labels=node.value
                plt.text(x, y,f"{labels} \n{node.count}\n" , ha='center', va='center', fontsize=7.5)

                 # Menggambar daun sebagai belah ketupat
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
                    plt.plot([x, left_x], [y - node_height/2, left_y + node_height/2], '-', color='grey')  # Penyesuaian koordinat y-akhir

                    # # Tambahkan teks di atas garis
                    # if node.feature_name in self.feature_encoders:
                    #     split_label = get_split_label(node.threshold, node.feature_name, self.feature_encoders, "left")
                    # else:
                    #      split_label = f' <= {node.threshold}'
                    plt.text((x + left_x) / 2 - (0.3*dx), y - node_height/2 - text_offset, f'{node.left_label}\n',ha='center', va='top', fontsize=8, color='red')

                    draw_node(ax,node.left, left_x, left_y, dx/2, dy/2,scale)

                if node.right is not None:
                    right_x = x + dx/2 
                    right_y = y - vertical_gap + leaf_height/2
                    plt.plot([x, right_x], [y - node_height/2, right_y+node_height/2], '-', color='grey')

                    # # Tambahkan teks di atas garis untuk cabang kanan (jika diperlukan)
                    # if node.feature_name in self.feature_encoders:
                    #     split_label = get_split_label(node.threshold, node.feature_name, self.feature_encoders,"right")
                        
                    # else :
                    #     split_label = f' > {node.threshold}'
                    plt.text((x + right_x ) / 2 + (0.3*dx), y - node_height/2 - text_offset, f'{node.right_label}',ha='center', va='top', fontsize=8, color='blue')
                    draw_node(ax, node.right, right_x, right_y, dx/2, dy/2,scale)
      

        # Hitung kedalaman maksimum pohon
        max_depth = self.calculate_max_depth(self.tree)
        max_depth=max_depth+1
        # print(max_depth)
        

        # Sesuaikan ukuran visualisasi berdasarkan kedalaman pohon
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
            plt.title("Decision Tree")
        plt.show()

    
    def calculate_max_depth(self, node):
        """
        Menghitung kedalaman maksimum dari pohon keputusan.

        Fungsi ini secara rekursif menghitung kedalaman maksimum dari pohon keputusan,
        yang bermanfaat untuk visualisasi dan analisis pohon.

        Args:
            node: Node saat ini yang sedang dianalisis.

        Returns:
            Kedalaman maksimum dari pohon.
        """
        if node is None or node.value is not None:
            return 0
        return 1 + max(self.calculate_max_depth(node.left), self.calculate_max_depth(node.right))

    