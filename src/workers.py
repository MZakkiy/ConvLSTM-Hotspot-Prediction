import numpy as np
from PySide6.QtCore import QThread, Signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dropout, Conv2D, Lambda, TimeDistributed
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import Constant

from .ml_core import weighted_binary_crossentropy, buat_metrik_spasial, SliceSequence, hitung_jarak_meleset_piksel, focal_loss

class TrainingWorker(QThread):
    update_progress = Signal(int)
    update_status = Signal(str)
    update_metrics = Signal(int, float, float) 
    training_finished = Signal()

    sinyal_evaluasi = Signal(float, float, float, float, float)

    def __init__(self, epochs, batch_size, train_gen, val_gen, layers, filters, dropout, optimizer, loss_func, eval_threshold, w0=1.0, w1=1.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.eval_threshold = eval_threshold
        
        self.num_layers = layers
        self.filters = filters
        self.dropout_rate = dropout
        self.optimizer_name = optimizer
        self.loss_name = loss_func

        self.w0 = w0
        self.w1 = w1
        self.focal_alpha = alpha
        self.focal_gamma = gamma

        self.horizon = getattr(self.train_gen, 'horizon', 1)
        self.time_steps = getattr(self.train_gen, 'time_steps', 7)

    def run(self):
        try:
            print(2)
            X_sample, Y_sample = self.train_gen[0]
            _, time_steps, tinggi, lebar, channels = X_sample.shape
            
            self.update_status.emit(f"Building Model: {self.num_layers} Layer, {self.filters} Filters...")
            
            model = Sequential()
            
            if self.horizon > self.time_steps:
                raise ValueError(f"Horizon ({self.horizon}) cannot be greater than time_steps ({time_steps})")

            for i in range(self.num_layers):
                ret_seq = True 
                
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
                
                if self.dropout_rate > 0:
                    model.add(Dropout(self.dropout_rate))
            
            model.add(TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same', bias_initializer=Constant(-4.5))))
            
            if self.loss_name == "Weighted Binary Crossentropy":
                loss_dipakai = weighted_binary_crossentropy(self.w0, self.w1)
            elif self.loss_name == "Focal Loss":
                loss_dipakai = focal_loss(alpha=self.focal_alpha, gamma=self.focal_gamma)
                
            model.compile(optimizer=self.optimizer_name, loss=loss_dipakai, metrics=['accuracy'])
            
            gui_callback = KerasWorkerCallback(self.update_progress, self.update_status, self.update_metrics, self.epochs)
        
            if self.val_gen is not None:
                model.fit(
                    self.train_gen,            
                    validation_data=self.val_gen, 
                    epochs=self.epochs,
                    callbacks=[gui_callback],
                    verbose=0
                )
            else:
                model.fit(
                    self.train_gen,            
                    epochs=self.epochs,
                    callbacks=[gui_callback],
                    verbose=0
                )
            
            self.model_hasil = model

            self.update_status.emit("Evaluate Model on Validation Data...")

            f_prec, f_rec, f_f1, f_auc = buat_metrik_spasial(self.eval_threshold)
            model.compile(optimizer='adam', loss='mse', metrics=[f_prec, f_rec, f_f1, f_auc])
            
            skor_evaluasi = model.evaluate(self.val_gen, verbose=0)
            
            val_precision = skor_evaluasi[1]
            val_recall = skor_evaluasi[2]
            val_f1 = skor_evaluasi[3]
            val_auc = skor_evaluasi[4]

            jarak_total = []
            
            if self.val_gen is not None:
                for i in range(len(self.val_gen)):
                    X_batch, y_batch = self.val_gen[i]
                    y_pred_batch = model.predict(X_batch, verbose=0)
                    
                    y_pred_biner = (y_pred_batch > self.eval_threshold).astype(int)
                    
                    for b in range(y_batch.shape[0]):
                        for t in range(y_batch.shape[1]):
                            jarak = hitung_jarak_meleset_piksel(y_batch[b, t, :, :, 0], y_pred_biner[b, t, :, :, 0])
                            if not np.isnan(jarak): 
                                jarak_total.append(jarak)
                                
            val_jarak = np.mean(jarak_total) if len(jarak_total) > 0 else 0.0

            self.sinyal_evaluasi.emit(val_precision, val_recall, val_f1, val_auc, val_jarak)
            
            self.update_status.emit("Training and evaluation done")
            
            self.training_finished.emit()
            
        except Exception as e:
            self.update_status.emit(f"Error Training: {str(e)}")
            self.training_finished.emit()

class EvaluasiWorker(QThread):
    sinyal_hasil = Signal(float, float, float, float, float)
    sinyal_status = Signal(str)
    
    def __init__(self, model_obj, val_gen, threshold):
        super().__init__()
        self.model = model_obj
        self.val_gen = val_gen
        self.threshold = threshold
        
    def run(self):
        try:
            self.sinyal_status.emit(f"Evaluating with Threshold {self.threshold:.2f}...")
            
            f_precision, f_recall, f_f1, f_auc = buat_metrik_spasial(self.threshold)
            
            self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy', f_precision, f_recall, f_f1, f_auc])

            skor = self.model.evaluate(self.val_gen, verbose=0)

            jarak_total = []
            
            if self.val_gen is not None:
                for i in range(len(self.val_gen)):
                    X_batch, y_batch = self.val_gen[i]
                    y_pred_batch = self.model.predict(X_batch, verbose=0)
                    
                    y_pred_biner = (y_pred_batch > self.threshold).astype(int)
                    
                    for b in range(y_batch.shape[0]):
                        for t in range(y_batch.shape[1]):
                            jarak = hitung_jarak_meleset_piksel(y_batch[b, t, :, :, 0], y_pred_biner[b, t, :, :, 0])
                            if not np.isnan(jarak):
                                jarak_total.append(jarak)
                                
            val_jarak = np.mean(jarak_total) if len(jarak_total) > 0 else 0.0

            self.sinyal_hasil.emit(skor[2], skor[3], skor[4], skor[5], val_jarak)
            self.sinyal_status.emit("Evaluation Complete!")
            
        except Exception as e:
            self.sinyal_status.emit(f"Error during Evaluation: {str(e)}")

class KerasWorkerCallback(Callback):
    def __init__(self, progress_signal, status_signal, metrics_signal, total_epochs):
        super().__init__()
        self.progress_signal = progress_signal
        self.status_signal = status_signal
        self.metrics_signal = metrics_signal 
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0.0)
        
        val_loss = logs.get('val_loss', 0.0)
        
        persentase = int(((epoch + 1) / self.total_epochs) * 100)
        self.progress_signal.emit(persentase)
        self.status_signal.emit(f"Epoch {epoch + 1}/{self.total_epochs} Complete | Loss: {loss:.4f}")
        
        self.metrics_signal.emit(epoch + 1, loss, val_loss)
