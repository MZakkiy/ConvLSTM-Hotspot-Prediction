import numpy as np
import math
from scipy.ndimage import zoom
import tensorflow as tf

class FireDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_hujan, data_suhu, data_kelem, data_hotspot, time_steps, horizon, batch_size, shuffle=True):
        self.hujan = data_hujan
        self.suhu = data_suhu
        self.kelem = data_kelem
        self.hotspot = data_hotspot
        self.time_steps = time_steps
        self.horizon = horizon 
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.indices = np.arange(len(self.hujan) - self.time_steps - self.horizon + 1)
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch, Y_batch = [], []
        
        for i in batch_indices:
            h_slice = self.hujan[i : i + self.time_steps]
            s_slice = self.suhu[i : i + self.time_steps]
            k_slice = self.kelem[i : i + self.time_steps]
            
            x_sample = np.stack([h_slice, s_slice, k_slice], axis=-1)
            X_batch.append(x_sample)
            
            y_sample = self.hotspot[i + self.time_steps : i + self.time_steps + self.horizon]
            
            y_sample = np.expand_dims(y_sample, axis=-1)
            Y_batch.append(y_sample)
            
        return np.array(X_batch), np.array(Y_batch)

    def on_epoch_end(self):
        """Every epoch, shuffle the indices if needed."""
        if self.shuffle:
            np.random.shuffle(self.indices)

def siapkan_data_mentah(data_hujan, data_suhu, data_kelem, df_hotspot, waktu_kordinat, extent_peta):
    max_hujan = np.nanmax(data_hujan) if np.any(~np.isnan(data_hujan)) else 0.0
    hujan_bersih = np.nan_to_num(data_hujan, nan=max_hujan)

    min_suhu = np.nanmin(data_suhu) if np.any(~np.isnan(data_suhu)) else 0.0
    suhu_bersih = np.nan_to_num(data_suhu, nan=min_suhu)

    kelem_bersih = np.nan_to_num(data_kelem, nan=1.0)

    min_hari = min(hujan_bersih.shape[0], suhu_bersih.shape[0], kelem_bersih.shape[0])
    hujan_bersih = hujan_bersih[:min_hari]
    suhu_bersih = suhu_bersih[:min_hari]
    kelem_bersih = kelem_bersih[:min_hari]
    waktu_kordinat = waktu_kordinat[:min_hari]

    tinggi_hujan, lebar_hujan = hujan_bersih.shape[1], hujan_bersih.shape[2]
    tinggi_suhu, lebar_suhu = suhu_bersih.shape[1], suhu_bersih.shape[2]
    tinggi_kelem, lebar_kelem = kelem_bersih.shape[1], kelem_bersih.shape[2]

    tinggi_target = max(tinggi_hujan, tinggi_suhu, tinggi_kelem)
    lebar_target = max(lebar_hujan, lebar_suhu, lebar_kelem)
    
    def resize_array_3d(arr):
        if arr.shape[1:] == (tinggi_target, lebar_target): 
            return arr
        fy = tinggi_target / arr.shape[1]
        fx = lebar_target / arr.shape[2]
        return zoom(arr, (1.0, fy, fx), order=1)

    hujan_bersih = resize_array_3d(hujan_bersih)
    suhu_bersih = resize_array_3d(suhu_bersih)
    kelem_bersih = resize_array_3d(kelem_bersih)

    Y_hotspot = np.zeros((min_hari, tinggi_target, lebar_target)) # Hemat memori, 3D dulu
    min_lon, max_lon, min_lat, max_lat = extent_peta
    col_date = 'acq_date' if 'acq_date' in df_hotspot.columns else 'date'
    
    for i, tanggal in enumerate(waktu_kordinat):
        api_hari_ini = df_hotspot[df_hotspot[col_date].dt.date == tanggal.date()]
        for _, row in api_hari_ini.iterrows():
            lon, lat = row['longitude'], row['latitude']
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                idx_x = int((lon - min_lon) / (max_lon - min_lon) * (lebar_target - 1))
                idx_y = int((max_lat - lat) / (max_lat - min_lat) * (tinggi_target - 1))
                Y_hotspot[i, idx_y, idx_x] = 1.0

    return hujan_bersih, suhu_bersih, kelem_bersih, Y_hotspot