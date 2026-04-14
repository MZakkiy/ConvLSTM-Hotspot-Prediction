import tensorflow as tf
from tensorflow.keras import backend as K

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
        y_true_expanded = tf.nn.max_pool2d(y_true_4d, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        true_positives = tf.reduce_sum(y_true_expanded * y_pred_biner)
        predicted_positives = tf.reduce_sum(y_pred_biner)
        return true_positives / (predicted_positives + tf.keras.backend.epsilon())

    def spatial_recall(y_true, y_pred):
        y_true_4d, y_pred_4d = siapkan_tensor(y_true, y_pred)
        
        y_pred_biner = tf.cast(y_pred_4d > batas_threshold, tf.float32)
        y_pred_expanded = tf.nn.max_pool2d(y_pred_biner, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        true_positives = tf.reduce_sum(y_true_4d * y_pred_expanded)
        actual_positives = tf.reduce_sum(y_true_4d)
        return true_positives / (actual_positives + tf.keras.backend.epsilon())

    def spatial_f1(y_true, y_pred):
        p = spatial_precision(y_true, y_pred)
        r = spatial_recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def spatial_auc(y_true, y_pred):
        y_true_4d, y_pred_4d = siapkan_tensor(y_true, y_pred)
        y_true_expanded = tf.nn.max_pool2d(y_true_4d, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        auc_obj.update_state(y_true_expanded, y_pred_4d)
        return auc_obj.result()

    spatial_precision.__name__ = 'spatial_precision'
    spatial_recall.__name__ = 'spatial_recall'
    spatial_f1.__name__ = 'spatial_f1'
    spatial_auc.__name__ = 'spatial_auc'
    
    return spatial_precision, spatial_recall, spatial_f1, spatial_auc