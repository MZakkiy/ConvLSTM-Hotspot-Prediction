import numpy as np
from PySide6.QtCore import QThread, Signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dropout, Conv2D, Lambda, TimeDistributed
from tensorflow.keras.callbacks import Callback

from .ml_core import weighted_binary_crossentropy, buat_metrik_spasial, SliceSequence, hitung_jarak_meleset_piksel

class TrainingWorker(QThread):
    update_progress = Signal(int)
    update_status = Signal(str)
    update_metrics = Signal(int, float, float) 
    training_finished = Signal()

    sinyal_evaluasi = Signal(float, float, float, float)

    # Tambahkan parameter X_data dan Y_data
    def __init__(self, epochs, batch_size, train_gen, val_gen, layers, filters, dropout, optimizer, loss_func, eval_threshold):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.eval_threshold = eval_threshold
        
        # Simpan Hyperparameter
        self.num_layers = layers
        self.filters = filters
        self.dropout_rate = dropout
        self.optimizer_name = optimizer
        self.loss_name = loss_func

        self.horizon = getattr(self.train_gen, 'horizon', 1)
        self.time_steps = getattr(self.train_gen, 'time_steps', 7)

    def run(self):
        try:
            X_sample, Y_sample = self.train_gen[0]
            _, time_steps, tinggi, lebar, channels = X_sample.shape
            
            self.update_status.emit(f"Membangun Model: {self.num_layers} Layer, {self.filters} Filters...")
            
            model = Sequential()
            
            # ==========================================
            # PEMBENTUKAN LAYER DINAMIS BERDASARKAN INPUT
            # ==========================================
            if self.horizon > self.time_steps:
                raise ValueError(f"Horizon ({self.horizon}) tidak boleh lebih besar dari time_steps ({time_steps})")

            for i in range(self.num_layers):
                # 🌟 PERUBAHAN 1: Semua ConvLSTM HARUS return_sequences=True
                ret_seq = True 
                
                # Layer pertama butuh input_shape
                if i == 0:
                    model.add(ConvLSTM2D(filters=self.filters, kernel_size=(3, 3), padding='same', 
                                        return_sequences=ret_seq, activation='relu', 
                                        input_shape=(time_steps, tinggi, lebar, channels)))
                else:
                    model.add(ConvLSTM2D(filters=self.filters, kernel_size=(3, 3), padding='same', 
                                        return_sequences=ret_seq, activation='relu'))
                    
                if i == self.num_layers - 1:
                    model.add(SliceSequence(horizon=self.horizon))
                
                model.add(BatchNormalization())
                
                # Tambahkan Dropout jika nilainya > 0
                if self.dropout_rate > 0:
                    model.add(Dropout(self.dropout_rate))
            
            # 🌟 PERUBAHAN 3: Layer Output dengan TimeDistributed
            # Conv2D memproses setiap hari secara independen namun tetap dalam bentuk sequence
            model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')))
            
            # ==========================================
            # PILIH FUNGSI LOSS BERDASARKAN COMBOBOX
            # ==========================================
            if self.loss_name == "Weighted Binary Crossentropy":
                # 1. Ambil matriks data target (hotspot) dari generator
                y_train_data = self.train_gen.hotspot
                
                # 2. Hitung jumlah pixel untuk masing-masing kelas
                total_api = np.sum(y_train_data)
                total_pixel = y_train_data.size
                total_aman = total_pixel - total_api
                
                # 3. Tetapkan bobot
                weight_zero = 1.0
                
                # Bobot kelas api adalah rasio kelas negatif (aman) dibagi positif (api)
                # Diberi kondisi if untuk mencegah error pembagian dengan nol (ZeroDivisionError)
                if total_api > 0:
                    weight_one = float(total_aman / total_api)
                else:
                    weight_one = 1.0 
                    
                # Kirim informasi bobot dinamis ini agar tampil di antarmuka GUI (Opsional)
                self.update_status.emit(f"Menerapkan bobot dinamis: Aman={weight_zero}, Api={weight_one:.2f}")
                
                loss_dipakai = weighted_binary_crossentropy(weight_zero, weight_one)
            elif self.loss_name == "MSE":
                loss_dipakai = 'mse'
            else:
                loss_dipakai = 'binary_crossentropy'
                
            model.compile(optimizer=self.optimizer_name, loss=loss_dipakai, metrics=['accuracy'])
            
            # 4. SAMBUNGKAN CALLBACK KE GUI
            gui_callback = KerasWorkerCallback(self.update_progress, self.update_status, self.update_metrics, self.epochs)
        
            # JIKA DATA VALIDASI TERSEDIA
            if self.val_gen is not None:
                model.fit(
                    self.train_gen,            
                    validation_data=self.val_gen, 
                    epochs=self.epochs,
                    callbacks=[gui_callback],
                    verbose=0
                )
            # JIKA DATANYA SEDIKIT DAN HANYA ADA TRAIN GEN
            else:
                model.fit(
                    self.train_gen,            
                    epochs=self.epochs,
                    callbacks=[gui_callback],
                    verbose=0
                )
            
            # 6. SIMPAN MODEL KE DALAM VARIABEL WORKER (BUKAN KE HARD DISK)
            self.model_hasil = model

            # ==========================================
            # EVALUASI OTOMATIS PADA DATA VALIDASI
            # ==========================================
            self.update_status.emit("Melakukan Evaluasi Akhir pada Data Validasi...")

            f_prec, f_rec, f_f1, f_auc = buat_metrik_spasial(self.eval_threshold)
            model.compile(optimizer='adam', loss='mse', metrics=[f_prec, f_rec, f_f1, f_auc])
            
            # Perintah ini akan menguji model pada data yang tidak dipakai belajar (val_gen)
            skor_evaluasi = model.evaluate(self.val_gen, verbose=0)
            
            # Urutan output dari evaluate() mengikuti urutan saat kita melakukan model.compile()
            # Yaitu: [loss, precision, recall, f1_metric]
            val_precision = skor_evaluasi[1]
            val_recall = skor_evaluasi[2]
            val_f1 = skor_evaluasi[3]
            val_auc = skor_evaluasi[4]
            
            # Tembakkan sinyalnya ke MainWindow!
            self.sinyal_evaluasi.emit(val_precision, val_recall, val_f1, val_auc)
            
            self.update_status.emit("Training dan Evaluasi Selesai!")
            
            self.update_status.emit("✅ Pelatihan Selesai! Model tersimpan di memori RAM.")
            self.training_finished.emit()
            
        except Exception as e:
            self.update_status.emit(f"❌ Error saat training: {str(e)}")
            self.training_finished.emit()

class EvaluasiWorker(QThread):
    sinyal_hasil = Signal(float, float, float, float)
    sinyal_status = Signal(str)
    
    # 🌟 UBAH path_model MENJADI model_obj
    def __init__(self, model_obj, val_gen, threshold):
        super().__init__()
        self.model = model_obj
        self.val_gen = val_gen
        self.threshold = threshold
        
    def run(self):
        try:
            # 🌟 HAPUS baris load_model dari .h5
            self.sinyal_status.emit(f"Menghitung ulang dengan Threshold {self.threshold:.2f}...")
            
            f_precision, f_recall, f_f1, f_auc = buat_metrik_spasial(self.threshold)
            
            # Langsung compile ulang model yang ada di RAM
            self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy', f_precision, f_recall, f_f1, f_auc])

            # for m in self.model.metrics:
            #     # Jika ini adalah kontainer 'CompileMetrics'
            #     if hasattr(m, 'metrics'):
            #         for sub_metric in m.metrics:
            #             if 'auc' in sub_metric.name.lower():
            #                 sub_metric.reset_state()
            #                 print(f"Berhasil meriset: {sub_metric.name}")
                
            #     # Jika metriknya berdiri sendiri (bergantung versi TF)
            #     elif 'auc' in m.name.lower():
            #         m.reset_state()
            #         print(f"Berhasil meriset: {m.name}")

            skor = self.model.evaluate(self.val_gen, verbose=0)
            
            self.sinyal_hasil.emit(skor[2], skor[3], skor[4], skor[5])
            self.sinyal_status.emit("Evaluasi Selesai!")
            
        except Exception as e:
            self.sinyal_status.emit(f"Error Evaluasi: {str(e)}")

class KerasWorkerCallback(Callback):
    # Tambahkan metrics_signal di sini
    def __init__(self, progress_signal, status_signal, metrics_signal, total_epochs):
        super().__init__()
        self.progress_signal = progress_signal
        self.status_signal = status_signal
        self.metrics_signal = metrics_signal 
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0.0)
        
        # Ambil val_loss jika ada (jika tidak ada kembalikan 0)
        val_loss = logs.get('val_loss', 0.0)
        
        persentase = int(((epoch + 1) / self.total_epochs) * 100)
        self.progress_signal.emit(persentase)
        self.status_signal.emit(f"Epoch {epoch + 1}/{self.total_epochs} Selesai | Loss: {loss:.4f}")
        
        # PANCARKAN SINYAL GRAFIK KE GUI
        self.metrics_signal.emit(epoch + 1, loss, val_loss)
