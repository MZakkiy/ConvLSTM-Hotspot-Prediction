import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from scipy.ndimage import distance_transform_edt


def weighted_binary_crossentropy(weight_zero, weight_one):
    """
    Fungsi Loss Kustom untuk mengatasi Class Imbalance pada peta.
    weight_zero: Bobot hukuman jika salah menebak area aman (kelas 0)
    weight_one: Bobot hukuman jika melewatkan titik api (kelas 1)
    """
    def loss(y_true, y_pred):
        # Pastikan tipe datanya seragam (float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Cegah nilai probabilitas menyentuh angka mutlak 0 atau 1 
        # (karena log(0) di matematika hasilnya tak terhingga / error)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Hitung hukuman untuk masing-masing tebakan
        loss_aman = weight_zero * (1. - y_true) * tf.math.log(1. - y_pred)
        loss_api = weight_one * y_true * tf.math.log(y_pred)
        
        # Kembalikan rata-rata total hukumannya
        return -tf.reduce_mean(loss_aman + loss_api)
        
    return loss

def buat_metrik_spasial(batas_threshold):
    auc_obj = tf.keras.metrics.AUC(name='spatial_auc')

    def siapkan_tensor(y_true, y_pred):
        """Fungsi pembantu untuk me-reshape 5D (B, T, H, W, 1) ke 4D (B*T, H, W, 1)"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Ambil ukuran dinamis dari tensor
        shape = tf.shape(y_true) 
        tinggi = shape[2]
        lebar = shape[3]
        
        # Gabungkan dimensi Batch dan Time
        y_true_4d = tf.reshape(y_true, [-1, tinggi, lebar, 1])
        y_pred_4d = tf.reshape(y_pred, [-1, tinggi, lebar, 1])
        
        return y_true_4d, y_pred_4d

    def spatial_precision(y_true, y_pred):
        y_true_4d, y_pred_4d = siapkan_tensor(y_true, y_pred)
        
        y_pred_biner = tf.cast(y_pred_4d > batas_threshold, tf.float32)
        y_true_expanded = tf.nn.max_pool2d(y_true_4d, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        true_positives = tf.reduce_sum(y_true_expanded * y_pred_biner)
        predicted_positives = tf.reduce_sum(y_pred_biner)
        return true_positives / (predicted_positives + tf.keras.backend.epsilon())

    def spatial_recall(y_true, y_pred):
        y_true_4d, y_pred_4d = siapkan_tensor(y_true, y_pred)
        
        y_pred_biner = tf.cast(y_pred_4d > batas_threshold, tf.float32)
        y_pred_expanded = tf.nn.max_pool2d(y_pred_biner, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        true_positives = tf.reduce_sum(y_true_4d * y_pred_expanded)
        actual_positives = tf.reduce_sum(y_true_4d)
        return true_positives / (actual_positives + tf.keras.backend.epsilon())

    def spatial_f1(y_true, y_pred):
        p = spatial_precision(y_true, y_pred)
        r = spatial_recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def spatial_auc(y_true, y_pred):
        y_true_4d, y_pred_4d = siapkan_tensor(y_true, y_pred)
        y_true_expanded = tf.nn.max_pool2d(y_true_4d, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        auc_obj.update_state(y_true_expanded, y_pred_4d)
        return auc_obj.result()

    spatial_precision.__name__ = 'spatial_precision'
    spatial_recall.__name__ = 'spatial_recall'
    spatial_f1.__name__ = 'spatial_f1'
    spatial_auc.__name__ = 'spatial_auc'
    
    return spatial_precision, spatial_recall, spatial_f1, spatial_auc

class SliceSequence(tf.keras.layers.Layer):
    """
    Custom Layer untuk memotong sequence/waktu.
    Aman untuk disimpan dan dimuat ulang oleh Keras.
    """
    def __init__(self, horizon=1, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon

    def call(self, inputs):
        # Ambil N frame terakhir di dimensi ke-1 (Waktu)
        return inputs[:, -self.horizon:, :, :, :]

    def get_config(self):
        # Fungsi wajib agar Keras bisa menyimpan layer ini ke file .keras
        config = super().get_config()
        config.update({"horizon": self.horizon})
        return config

def hitung_jarak_meleset_piksel(y_true_peta, y_pred_peta):
    """
    Menghitung rata-rata jarak meleset (dalam piksel) dari prediksi titik api 
    terhadap titik api sebenarnya menggunakan Distance Transform.
    """
    # 1. Kasus ekstrem: Model tidak menebak ada api sama sekali (False Negatives total)
    if np.sum(y_pred_peta) == 0:
        return 0.0 
        
    # 2. Kasus ekstrem: Tidak ada api sama sekali di data asli, tapi model menebak ada api (False Positives total)
    if np.sum(y_true_peta) == 0:
        return np.nan 

    # 3. Balikkan nilai y_true (0 jadi 1, 1 jadi 0) karena SciPy menghitung jarak ke nilai 0
    titik_api_sebenarnya_inverted = (y_true_peta == 0).astype(int)

    # 4. Buat Peta Jarak (Distance Transform)
    peta_jarak = distance_transform_edt(titik_api_sebenarnya_inverted)

    # 5. Ambil nilai jarak HANYA di koordinat tempat model memprediksi ada api
    jarak_tebakan_model = peta_jarak[y_pred_peta == 1]

    # 6. Rata-ratakan jarak tersebut
    rata_rata_error_piksel = np.mean(jarak_tebakan_model)

    return rata_rata_error_piksel