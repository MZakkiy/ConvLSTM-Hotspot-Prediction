import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from scipy.ndimage import distance_transform_edt


def weighted_binary_crossentropy(weight_zero, weight_one):
    """
    Loss Kustom: Weighted Binary Crossentropy
    weight_zaro : Weight for the "safe" class (class 0)
    weight_one : Weight for the "fire" class (class 1)
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        loss_aman = weight_zero * (1. - y_true) * tf.math.log(1. - y_pred)
        loss_api = weight_one * y_true * tf.math.log(y_pred)
        
        return -tf.reduce_mean(loss_aman + loss_api)
        
    return loss

def buat_metrik_spasial(batas_threshold):
    auc_obj = tf.keras.metrics.AUC(name='spatial_auc')

    def siapkan_tensor(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        shape = tf.shape(y_true) 
        tinggi = shape[2]
        lebar = shape[3]
        
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
    Custom layer for slicing temporal sequences.
    """
    def __init__(self, horizon=1, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon

    def call(self, inputs):
        return inputs[:, -self.horizon:, :, :, :]

    def get_config(self):
        config = super().get_config()
        config.update({"horizon": self.horizon})
        return config

def hitung_jarak_meleset_piksel(y_true_peta, y_pred_peta):
    """
    Calculates the average pixel distance error between the predicted hotspot locations and the actual hotspot locations using Distance Transform.
    """
    if np.sum(y_pred_peta) == 0:
        return 0.0 
        
    if np.sum(y_true_peta) == 0:
        return np.nan 

    titik_api_sebenarnya_inverted = (y_true_peta == 0).astype(int)

    peta_jarak = distance_transform_edt(titik_api_sebenarnya_inverted)

    jarak_tebakan_model = peta_jarak[y_pred_peta == 1]

    rata_rata_error_piksel = np.mean(jarak_tebakan_model)

    return rata_rata_error_piksel

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Loss custom: Focal Loss
    - alpha: Balance between positive and negative classes.
    - gamma: Gives more focus to hard examples.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        bce_api = -y_true * tf.math.log(y_pred)
        bce_aman = -(1. - y_true) * tf.math.log(1. - y_pred)
        
        loss_api = alpha * tf.math.pow(1. - y_pred, gamma) * bce_api
        loss_aman = (1. - alpha) * tf.math.pow(y_pred, gamma) * bce_aman
        
        return tf.reduce_mean(loss_api + loss_aman)
        
    return loss