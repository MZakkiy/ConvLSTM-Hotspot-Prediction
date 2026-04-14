import numpy as np
import math
from scipy.ndimage import zoom
import xarray
import pandas
import tensorflow as tf

class FireDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_hujan, data_suhu, data_kelem, data_hotspot, time_steps, horizon, batch_size, shuffle=True):
        self.hujan = data_hujan
        self.suhu = data_suhu
        self.kelem = data_kelem
        self.hotspot = data_hotspot
        self.time_steps = time_steps
        self.horizon = horizon  # Jumlah hari prediksi ke depan
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Batasi indeks agar tidak mengambil data di luar batas array
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
            # 1. Ambil Input (7 Hari ke belakang)
            h_slice = self.hujan[i : i + self.time_steps]
            s_slice = self.suhu[i : i + self.time_steps]
            k_slice = self.kelem[i : i + self.time_steps]
            
            x_sample = np.stack([h_slice, s_slice, k_slice], axis=-1)
            X_batch.append(x_sample)
            
            # 2. Ambil Target (3 Hari ke depan, dimulai setelah time_steps)
            y_sample = self.hotspot[i + self.time_steps : i + self.time_steps + self.horizon]
            
            # Tambahkan dimensi channel di akhir: (Horizon, H, W) -> (Horizon, H, W, 1)
            y_sample = np.expand_dims(y_sample, axis=-1)
            Y_batch.append(y_sample)
            
        return np.array(X_batch), np.array(Y_batch)

    def on_epoch_end(self):
        """Setiap kali 1 Epoch selesai, acak ulang urutannya agar model tidak menghafal."""
        if self.shuffle:
            np.random.shuffle(self.indices)

def siapkan_data_mentah(data_hujan, data_suhu, data_kelem, df_hotspot, waktu_kordinat, extent_peta):
    """
    Hanya merapikan data menjadi 3D (Hari, Lat, Lon), TIDAK membuat Tensor 5D.
    """
    # 1. Bersihkan NaN
    hujan_bersih = np.nan_to_num(data_hujan, nan=0.0)
    suhu_bersih = np.nan_to_num(data_suhu, nan=0.0)
    kelem_bersih = np.nan_to_num(data_kelem, nan=0.0)

    # 2. Samakan Jumlah Hari
    min_hari = min(hujan_bersih.shape[0], suhu_bersih.shape[0], kelem_bersih.shape[0])
    hujan_bersih = hujan_bersih[:min_hari]
    suhu_bersih = suhu_bersih[:min_hari]
    kelem_bersih = kelem_bersih[:min_hari]
    waktu_kordinat = waktu_kordinat[:min_hari]

    # 3. Samakan Ukuran Spasial (Resize ke ukuran Hujan/CHIRPS)
    # Kita pakai fungsi zoom dari scipy
    tinggi_target, lebar_target = hujan_bersih.shape[1], hujan_bersih.shape[2]
    
    def resize_array_3d(arr):
        if arr.shape[1:] == (tinggi_target, lebar_target): return arr
        fy = tinggi_target / arr.shape[1]
        fx = lebar_target / arr.shape[2]
        return zoom(arr, (1.0, fy, fx), order=1)

    if suhu_bersih.shape != hujan_bersih.shape:
        suhu_bersih = resize_array_3d(suhu_bersih)
    if kelem_bersih.shape != hujan_bersih.shape:
        kelem_bersih = resize_array_3d(kelem_bersih)

    # 4. Rasterisasi Hotspot (Ubah CSV jadi Peta 0/1)
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

    # Kembalikan 4 bahan baku terpisah
    return hujan_bersih, suhu_bersih, kelem_bersih, Y_hotspot