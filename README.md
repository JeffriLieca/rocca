# Rocca: Simple Machine Learning

Rocca adalah library Python yang berfokus pada penyediaan implementasi berbagai algoritma machine learning, termasuk metode ensemble, clustering, dan association rule mining.

## Fitur

- Algoritma ensemble: Bagging, Random Forest, XGBoost
- Clustering: K-Means
- Association Rule Mining: Apriori
- Decision Tree: Standar Decision Tree dan Regression Tree

## Tujuan

Rocca dirancang untuk menyederhanakan proses pembelajaran dan eksplorasi dalam pembelajaran mesin. Dengan kode yang sederhana dan mudah dipahami, Rocca bertujuan untuk menjadi alat bantu yang efektif bagi mereka yang baru memulai atau sedang belajar tentang pembelajaran mesin.

## Instalasi

Untuk menginstal Rocca, cukup gunakan pip:

```bash
pip install rocca
```

## Penggunaan

Berikut adalah beberapa contoh penggunaan library Rocca:

```
from rocca.clustering import KMeans

model = KMeans(k=3)
model.fit(X)
labels = model.get_cluster_labels()
```

## Source Code

Akses source codenya di [GitHub](https://github.com/).
