import numpy as np

def sigmoid(x):
    """
    Menghitung nilai sigmoid dari input x.

    Fungsi sigmoid mengubah nilai input menjadi nilai antara 0 dan 1, 
    yang sering digunakan dalam konteks klasifikasi biner untuk menghasilkan probabilitas.

    Args:
        x (numeric or numpy.array): Nilai input.

    Returns:
        numeric or numpy.array: Nilai sigmoid dari x.
    """
    return 1 / (1 + np.exp(-x))

def calculate_log_odds(y):
    """
    Menghitung log odds dari target y.

    Log odds digunakan untuk menginisialisasi prediksi dalam konteks boosting,
    khususnya untuk kasus klasifikasi biner.

    Args:
        y (numpy.array): Target biner.

    Returns:
        float: Log odds dari y.
    """
    p = np.mean(y)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))

def log_loss(y_true, y_pred):
    """
    Menghitung log loss antara nilai prediksi dan nilai sebenarnya.

    Log loss mengukur performa model klasifikasi, di mana nilai yang lebih kecil menunjukkan performa yang lebih baik.

    Args:
        y_true (numpy.array): Nilai target sebenarnya.
        y_pred (numpy.array): Nilai prediksi.

    Returns:
        float: Log loss antara y_true dan y_pred.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
