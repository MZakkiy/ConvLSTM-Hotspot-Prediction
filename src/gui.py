import os

import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.ndimage import zoom
import xarray as xr
import rioxarray
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PySide6.QtWidgets import (QApplication, QLineEdit, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGroupBox, QComboBox, QTabWidget, 
                               QPushButton, QLabel, QSlider, QFileDialog, QMessageBox,
                               QProgressBar, QSpinBox, QDoubleSpinBox, QGridLayout, QFormLayout)
from PySide6.QtGui import QIcon, QPainter, QColor
from PySide6.QtCore import Qt

from sklearn.model_selection import train_test_split

from .data_handler import FireDataGenerator, siapkan_data_mentah
from .workers import TrainingWorker, EvaluasiWorker

from tensorflow.keras.models import load_model
from src.ml_core import weighted_binary_crossentropy, SliceSequence


# Import jembatan penghubung Matplotlib dan PySide
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MapCanvas(FigureCanvas):
    """Kelas khusus untuk menampung peta Matplotlib di dalam PySide"""
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # Setting background warna terang untuk tema terang
        self.fig.patch.set_facecolor('#ffffff') 
        self.axes.set_facecolor('#ffffff')
        self.axes.tick_params(colors='black')
        
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Variabel penampung data asli
        self.data_hujan = None
        self.data_suhu = None
        self.data_kelembapan = None
        self.df_hotspot = None
        self.waktu_kordinat = None
        # Variabel untuk menampung model ML di memori
        self.model_convlstm = None

        # Variabel penampung riwayat metrik
        self.history_epochs = []
        self.history_loss = []
        self.history_val_loss = []

        self.data_hujan = None
        self.data_suhu = None
        self.data_kelembapan = None
        self.df_hotspot = None
        self.waktu_kordinat = None

        self.extent_hujan = None
        self.extent_suhu = None
        self.extent_kelembapan = None

        self.config_region = {
            "Sumatra": {
                "extent": [95.0, 107.0, -6.0, 6.0], # [lon_min, lon_max, lat_min, lat_max]
                "shp_batas": "shapefiles/batas_sumatra.shp"
            },
            "Kalimantan": {
                "extent": [108.0, 120.0, -5.0, 5.0], 
                "shp_batas": "shapefiles/batas_kalimantan.shp" 
            }
        }
        
        self.setWindowTitle("Sistem Prediksi Titik Panas Lahan Gambut")
        self.resize(1000, 700)
        
        # Setting Tema Terang
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f5f5f5; color: black; }
            QGroupBox { border: 1px solid #cccccc; border-radius: 5px; margin-top: 1ex; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QComboBox, QPushButton { background-color: #ffffff; border: 1px solid #cccccc; padding: 5px; color: black; }
            QTabWidget::pane { border: 1px solid #cccccc; }
            QTabBar::tab { background: #e0e0e0; padding: 8px; border: 1px solid #cccccc; color: black; }
            QTabBar::tab:selected { background: #ffffff; color: black; }
        """)

        # Widget Utama
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. BUAT TABS UTAMA
        self.tabs = QTabWidget()
        self.tab_data_prep = QWidget()
        self.tabs.addTab(self.tab_data_prep, "Data Preparation")
        main_layout.addWidget(self.tabs)

        # Layout untuk Tab 1 (Data Preparation)
        tab1_layout = QHBoxLayout(self.tab_data_prep)

        # 2. BUAT PANEL KONTROL ATAS (Horizontal)
        top_control_layout = QVBoxLayout()

        # --- KOTAK "Input Data" ---
        group_input = QGroupBox("Input Data")
        vbox_input = QVBoxLayout()

        self.combo_region = QComboBox()
        self.combo_region.addItems(["Sumatra", "Kalimantan"])
        self.combo_region.currentTextChanged.connect(self.ganti_region)
        
        self.combo_variable = QComboBox()
        self.combo_variable.addItems(["Rainfall", "Temperature", "Soil Moisture", "Hotspots"])
        self.combo_variable.currentIndexChanged.connect(self.update_map)
        
        # Tombol Import
        self.btn_import = QPushButton("Import Data")
        self.btn_import.clicked.connect(self.import_data) 
        
        vbox_input.addWidget(QLabel("Select Region:"))
        vbox_input.addWidget(self.combo_region)
        vbox_input.addWidget(QLabel("Variable"))
        vbox_input.addWidget(self.combo_variable)
        vbox_input.addWidget(self.btn_import)
        
        group_input.setLayout(vbox_input)
               
        top_control_layout.addWidget(group_input)
        top_control_layout.addStretch() 

        # 3. BUAT AREA MAIN PLOT (PETA SPASIAL)
        group_plot = QGroupBox("Main Plot (Spatial Map)")
        plot_layout = QVBoxLayout()
        
        # Inisialisasi Canvas Matplotlib
        self.canvas = MapCanvas(self, width=6, height=6, dpi=100)
        
        # Tambahkan Toolbar agar user bisa Zoom-in, Zoom-out, dan Pan di peta
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: transparent; /* Menyatu sempurna dengan warna UI bawaan */
                border: none;
            }
            QToolButton {
                background-color: transparent; /* Tombol toolbar berwarna transparent */
                color: #000000; /* Ikon toolbar berwarna hitam untuk kontras */
                border: none;
                border-radius: 4px;
                padding: 4px;
            }
            QToolButton:hover {
                background-color: rgba(0, 0, 0, 0.1); /* Efek abu-abu elegan saat disorot mouse */
            }
            QToolButton:pressed {
                background-color: rgba(0, 0, 0, 0.2); /* Efek sedikit lebih gelap saat diklik */
            }
            QLabel {
                color: #000000; /* Teks koordinat (x,y) dkk menjadi hitam */
                background-color: transparent;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                padding-right: 10px;
            }
        """)

        for action in self.toolbar.actions():
            icon_lama = action.icon()
            
            # Pastikan action ini punya ikon (bukan teks atau spasi kosong)
            if not icon_lama.isNull():
                # Ambil gambar ikon dalam ukuran 24x24 pixel
                pixmap = icon_lama.pixmap(24, 24)
                
                # Buat "kuas" untuk mengecat di atas gambar tersebut
                painter = QPainter(pixmap)
                
                # Mode ini penting: Hanya cat area yang BUKAN transparan (bentuk ikonnya saja)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                
                # Timpa dengan warna hitam pekat
                painter.fillRect(pixmap.rect(), QColor("#000000")) 
                painter.end() # Selesai mengecat
                
                # Pasang kembali ikon yang sudah dihitamkan ke tombolnya
                action.setIcon(QIcon(pixmap))
        # -----------------------------------------------------


        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        # 2. Tambahkan Time Slider di Bawah Peta
        slider_layout = QHBoxLayout()
        
        label_teks_waktu = QLabel("Slide Time:")
        label_teks_waktu.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.slider_waktu = QSlider(Qt.Horizontal)
        self.slider_waktu.setMinimum(0)             # Indeks awal (Hari ke-1)
        #self.slider_waktu.setMaximum(TOTAL_HARI - 1) # Indeks akhir (Hari ke-30)
        self.slider_waktu.setValue(0)
        # Hubungkan aksi geser slider dengan fungsi update peta
        self.slider_waktu.valueChanged.connect(self.update_map)
        
        self.label_hari_aktif = QLabel("Day 1")
        self.label_hari_aktif.setStyleSheet("font-weight: bold; color: #ff6600; font-size: 14px;")
        
        slider_layout.addWidget(label_teks_waktu)
        slider_layout.addWidget(self.slider_waktu)
        slider_layout.addWidget(self.label_hari_aktif)
        
        plot_layout.addLayout(slider_layout)
        group_plot.setLayout(plot_layout)
        
        tab1_layout.addLayout(top_control_layout)
        tab1_layout.addWidget(group_plot, stretch=1)

        # Tampilkan peta awal saat aplikasi dibuka
        self.update_map()

        # ==========================================
        # TAB 2: BUILD ML MODEL
        # ==========================================
        self.tab_ml_model = QWidget()
        self.tabs.addTab(self.tab_ml_model, "Build ML Model")
        
        layout_ml = QHBoxLayout(self.tab_ml_model)
        
        # --- PANEL KIRI: PENGATURAN HYPERPARAMETER ---
        panel_kiri_ml = QVBoxLayout()
        
        # --- GRUP 1: MODEL ARCHITECTURE ---
        group_arsitektur = QGroupBox("Model Architecture (ConvLSTM)")
        grid_arsitektur = QGridLayout()
        
        grid_arsitektur.addWidget(QLabel("Time Steps (Day):"), 0, 0)
        self.spin_timesteps = QSpinBox()
        self.spin_timesteps.setRange(1, 30)
        self.spin_timesteps.setValue(7)
        grid_arsitektur.addWidget(self.spin_timesteps, 0, 1)
        
        grid_arsitektur.addWidget(QLabel("Hidden Layers:"), 1, 0)
        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(1, 5)
        self.spin_layers.setValue(1)
        grid_arsitektur.addWidget(self.spin_layers, 1, 1)
        
        grid_arsitektur.addWidget(QLabel("Filters (Neurons):"), 0, 2)
        self.spin_filters = QSpinBox()
        self.spin_filters.setRange(8, 128)
        self.spin_filters.setSingleStep(8) # Naik turun kelipatan 8
        self.spin_filters.setValue(16)
        grid_arsitektur.addWidget(self.spin_filters, 0, 3)
        
        grid_arsitektur.addWidget(QLabel("Dropout Rate:"), 1, 2)
        self.spin_dropout = QDoubleSpinBox() # Untuk angka desimal
        self.spin_dropout.setRange(0.0, 0.9)
        self.spin_dropout.setSingleStep(0.1)
        self.spin_dropout.setValue(0.2)
        grid_arsitektur.addWidget(self.spin_dropout, 1, 3)

        grid_arsitektur.addWidget(QLabel("Days To Predict:"), 2, 0)
        self.spin_horizon = QSpinBox()
        self.spin_horizon.setRange(1, 7) # Misal: 1 sampai 7 hari
        self.spin_horizon.setValue(3)    # Default 3 hari
        grid_arsitektur.addWidget(self.spin_horizon, 2, 1)
        
        group_arsitektur.setLayout(grid_arsitektur)
        panel_kiri_ml.addWidget(group_arsitektur)
        
        # --- GRUP 2: MODEL TRAINING ---
        group_training = QGroupBox("Model Training")
        grid_training = QGridLayout()
        
        grid_training.addWidget(QLabel("Epochs:"), 0, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(10)
        grid_training.addWidget(self.spin_epochs, 0, 1)
        
        grid_training.addWidget(QLabel("Batch Size:"), 1, 0)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(8)
        grid_training.addWidget(self.spin_batch, 1, 1)
        
        grid_training.addWidget(QLabel("Optimizer:"), 0, 2)
        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(["adam", "rmsprop", "sgd"])
        grid_training.addWidget(self.combo_optimizer, 0, 3)
        
        grid_training.addWidget(QLabel("Loss Function:"), 1, 2)
        self.combo_loss = QComboBox()
        self.combo_loss.addItems(["Weighted Binary Crossentropy", "Binary Crossentropy", "MSE"])
        grid_training.addWidget(self.combo_loss, 1, 3)
        
        group_training.setLayout(grid_training)
        panel_kiri_ml.addWidget(group_training)

        group_eval = QGroupBox("Model Evaluation")
        form_eval = QFormLayout()

        self.label_val_auc = QLineEdit("N/A")
        self.label_val_auc.setReadOnly(True)
        self.label_val_auc.setAlignment(Qt.AlignCenter)
        
        form_eval.addRow("ROC-AUC:", self.label_val_auc)

        self.spin_eval_threshold = QDoubleSpinBox()
        self.spin_eval_threshold.setRange(0.01, 0.99)
        self.spin_eval_threshold.setSingleStep(0.05)
        self.spin_eval_threshold.setValue(0.50)
        self.spin_eval_threshold.setToolTip("The probability threshold for recognizing a hot spot during evaluation. Adjusting this can affect precision and recall metrics.")
        
        form_eval.addRow("Eval Threshold:", self.spin_eval_threshold) # Masukkan ke layout evaluasi

        self.btn_re_eval = QPushButton("Re-evaluate with New Threshold")
        self.btn_re_eval.clicked.connect(self.jalankan_evaluasi_cepat)
        form_eval.addRow("", self.btn_re_eval) # Kosongkan label kiri agar rapi
        
        self.label_val_precision = QLineEdit("N/A")
        self.label_val_precision.setReadOnly(True)
        
        self.label_val_recall = QLineEdit("N/A")
        self.label_val_recall.setReadOnly(True)
        
        self.label_val_f1 = QLineEdit("N/A")
        self.label_val_f1.setReadOnly(True)
        
        form_eval.addRow("Precision:", self.label_val_precision)
        form_eval.addRow("Recall:", self.label_val_recall)
        form_eval.addRow("F1-Score:", self.label_val_f1)
        
        group_eval.setLayout(form_eval)
        panel_kiri_ml.addWidget(group_eval)
        
        # --- TOMBOL MULAI & PROGRESS ---
        self.progress_bar = QProgressBar() 
        self.progress_bar.setValue(0)

        self.btn_train = QPushButton("Start Training") 
        self.btn_train.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 5px;") 
        self.btn_train.clicked.connect(self.mulai_training) 

        self.btn_simpan_model = QPushButton("Save Model")
        self.btn_simpan_model.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 5px;")
        self.btn_simpan_model.setEnabled(False) # Matikan tombol sebelum ada model yang dilatih
        self.btn_simpan_model.clicked.connect(self.simpan_model)

        panel_kiri_ml.addWidget(self.btn_train)
        panel_kiri_ml.addWidget(self.btn_simpan_model)
        panel_kiri_ml.addWidget(self.progress_bar)
        panel_kiri_ml.addStretch()
        
        # --- PANEL KANAN: GRAFIK REAL-TIME ---
        group_grafik = QGroupBox("Live Training Metrics (Loss & Validation)")
        layout_grafik = QVBoxLayout()
        
        # Buat Canvas Matplotlib baru khusus untuk grafik garis
        self.fig_metrics = Figure(figsize=(6, 4), dpi=100)
        self.fig_metrics.patch.set_facecolor('#ffffff')
        self.ax_metrics = self.fig_metrics.add_subplot(111)
        self.ax_metrics.set_facecolor('#ffffff')
        self.ax_metrics.tick_params(colors='black')
        self.ax_metrics.set_xlabel('Epoch', color='black')
        self.ax_metrics.set_ylabel('Loss', color='black')
        
        self.canvas_metrics = FigureCanvas(self.fig_metrics)
        layout_grafik.addWidget(self.canvas_metrics)
        group_grafik.setLayout(layout_grafik)
        
        # Gabungkan panel kiri dan kanan
        layout_ml.addLayout(panel_kiri_ml, stretch=1)
        layout_ml.addWidget(group_grafik, stretch=3)
        
        # ==========================================
        # TAB 3: FIRE INDEX (PREDIKSI)
        # ==========================================
        self.tab_fire_index = QWidget()
        self.tabs.addTab(self.tab_fire_index, "Fire Index")
        
        layout_fire = QHBoxLayout(self.tab_fire_index)
        
        # --- PANEL KIRI: KONTROL PREDIKSI ---
        panel_kiri_fire = QVBoxLayout()
        group_kontrol_fire = QGroupBox("Control Panel")
        layout_kontrol = QVBoxLayout()

        self.btn_muat_model = QPushButton("Load Pre-trained Model")
        self.btn_muat_model.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; margin-bottom: 10px; padding: 10px;")
        self.btn_muat_model.clicked.connect(self.muat_model)
        layout_kontrol.addWidget(self.btn_muat_model)
        
        self.btn_prediksi = QPushButton("Predict Fire Risk")
        self.btn_prediksi.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; font-size: 14px; padding: 15px; border-radius: 5px;")
        self.btn_prediksi.clicked.connect(self.jalankan_prediksi)
        layout_kontrol.addWidget(self.btn_prediksi)

        self.form_prediksi = QFormLayout()

        self.slider_threshold = QDoubleSpinBox()
        self.slider_threshold.setRange(0.01, 0.99)
        self.slider_threshold.setSingleStep(0.01)
        self.slider_threshold.setValue(0.30)
        self.slider_threshold.setDecimals(2)
        self.slider_threshold.valueChanged.connect(lambda: self.update_peta_prediksi(self.slider_hari.value()))

        self.form_prediksi.addRow("Confidence Threshold:", self.slider_threshold)
        layout_kontrol.addLayout(self.form_prediksi)
        
        group_kontrol_fire.setLayout(layout_kontrol)
        panel_kiri_fire.addWidget(group_kontrol_fire)
        panel_kiri_fire.addStretch()
        
        # --- PANEL KANAN: PETA HASIL PREDIKSI ---
        group_peta_fire = QGroupBox("Predicted Fire Risk Map")
        layout_peta_fire = QVBoxLayout()

        self.canvas_prediksi = MapCanvas(self, width=6, height=6, dpi=100) # Canvas khusus untuk peta prediksi
        self.canvas_prediksi.axes.set_title("No prediction has been run yet", color='black', pad=10)

        self.toolbar_prediksi = NavigationToolbar(self.canvas_prediksi, self)
        self.toolbar_prediksi.setStyleSheet("""
            QToolBar {
                background-color: transparent; /* Menyatu sempurna dengan warna UI bawaan */
                border: none;
            }
            QToolButton {
                background-color: transparent; /* Tombol toolbar berwarna transparent */
                color: #000000; /* Ikon toolbar berwarna hitam untuk kontras */
                border: none;
                border-radius: 4px;
                padding: 4px;
            }
            QToolButton:hover {
                background-color: rgba(0, 0, 0, 0.1); /* Efek abu-abu elegan saat disorot mouse */
            }
            QToolButton:pressed {
                background-color: rgba(0, 0, 0, 0.2); /* Efek sedikit lebih gelap saat diklik */
            }
            QLabel {
                color: #000000; /* Teks koordinat (x,y) dkk menjadi hitam */
                background-color: transparent;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                padding-right: 10px;
            }
        """)

        for action in self.toolbar_prediksi.actions():
            icon_lama = action.icon()
            
            # Pastikan action ini punya ikon (bukan teks atau spasi kosong)
            if not icon_lama.isNull():
                # Ambil gambar ikon dalam ukuran 24x24 pixel
                pixmap = icon_lama.pixmap(24, 24)
                
                # Buat "kuas" untuk mengecat di atas gambar tersebut
                painter = QPainter(pixmap)
                
                # Mode ini penting: Hanya cat area yang BUKAN transparan (bentuk ikonnya saja)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                
                # Timpa dengan warna hitam pekat
                painter.fillRect(pixmap.rect(), QColor("#000000")) 
                painter.end() # Selesai mengecat
                
                # Pasang kembali ikon yang sudah dihitamkan ke tombolnya
                action.setIcon(QIcon(pixmap))
        # -----------------------------------------------------

        self.slider_hari = QSlider(Qt.Horizontal)
        self.slider_hari.setMinimum(1)
        self.slider_hari.setMaximum(1) # Akan diupdate nanti setelah prediksi dijalankan
        self.slider_hari.setTickPosition(QSlider.TicksBelow)
        self.slider_hari.setTickInterval(1)
        self.slider_hari.setEnabled(False) # Matikan sebelum ada hasil prediksi

        self.slider_hari.valueChanged.connect(self.update_peta_prediksi)

        layout_peta_fire.addWidget(self.toolbar_prediksi)
        layout_peta_fire.addWidget(self.canvas_prediksi)
        layout_peta_fire.addWidget(self.slider_hari)
        group_peta_fire.setLayout(layout_peta_fire)
        
        # Gabungkan
        layout_fire.addLayout(panel_kiri_fire, stretch=1)
        layout_fire.addWidget(group_peta_fire, stretch=3)

    def ganti_region(self):
        """Mereset data jika pengguna mengganti pulau agar matriks tidak bentrok"""
        self.data_hujan = None
        self.data_suhu = None
        self.data_kelembapan = None
        self.df_hotspot = None
        self.waktu_kordinat = None
        self.update_map()

    
    def import_data(self):
        """Membuka jendela dialog dan memuat data asli ke dalam memori"""
        pilihan = self.combo_variable.currentText()
        pilihan_region = self.combo_region.currentText() # Ambil pilihan pulau
        
        # Ambil path shapefile sesuai pulau yang dipilih
        path_shp_aktif = self.config_region[pilihan_region]["shp_batas"]

        if "Hotspots" in pilihan:
            filter_file = "Data CSV (*.csv);;Semua File (*.*)"
            judul_dialog = "Select Hotspot CSV File"
        else:
            # Tambahkan .tif ke dalam filter
            filter_file = "Data Raster (*.nc *.tif *.tiff);;Semua File (*.*)"
            judul_dialog = f"Select Raster File for {pilihan}"

        file_path, _ = QFileDialog.getOpenFileName(self, judul_dialog, "", filter_file)
        
        if file_path:
            nama_file = os.path.basename(file_path)
            
            try:
                # --- 1. JIKA FILE ADALAH CSV TITIK PANAS ---
                if "Hotspots" in pilihan:
                    # Membaca data menggunakan Pandas
                    self.df_hotspot = pd.read_csv(file_path)
                    
                    # NASA FIRMS biasanya menggunakan nama kolom 'acq_date'
                    col_date = 'acq_date' if 'acq_date' in self.df_hotspot.columns else 'date'
                    
                    # Ubah format teks menjadi format Tanggal (Datetime)
                    # self.df_hotspot[col_date] = pd.to_datetime(self.df_hotspot[col_date])

                    self.df_hotspot[col_date] = pd.to_datetime(self.df_hotspot[col_date])
                    
                    # --- 🌟 TAMBAHAN: FILTER SPASIAL TITIK API DI LAHAN GAMBUT ---
                    try:
                        # Muat shapefile batas gambut
                        gdf_gambut = gpd.read_file("shapefiles/Indonesia_peat_lands.shp")
                        gdf_gambut = gdf_gambut.to_crs("EPSG:4326")
                        
                        # Jadikan dataframe pandas sebagai GeoDataFrame agar memiliki atribut spasial
                        gdf_api = gpd.GeoDataFrame(
                            self.df_hotspot, 
                            geometry=gpd.points_from_xy(self.df_hotspot['longitude'], self.df_hotspot['latitude']),
                            crs="EPSG:4326"
                        )
                        
                        # Potong titik api agar hanya menyisakan yang mendarat di dalam poligon gambut
                        gdf_api_gambut = gpd.clip(gdf_api, gdf_gambut)
                        
                        # Kembalikan ke wujud DataFrame biasa dan buang kolom geometri
                        self.df_hotspot = pd.DataFrame(gdf_api_gambut).drop(columns=['geometry'])
                    except Exception as e:
                        print(f"Error: Could not filter hotspots by peatland shapefile. Proceeding with all hotspots. Details: {e}")
                    # --------------------------------------------------------------
                    
                    # --- PERBAIKAN SINKRONISASI WAKTU ---
                    # Jika user memuat data NASA sebelum memuat data Suhu/Hujan,
                    # kita harus mengambil daftar tanggal unik dari CSV ini untuk mengaktifkan Slider
                    if self.waktu_kordinat is None:
                        # Ambil semua tanggal yang ada di file, urutkan dari yang tertua
                        tanggal_unik = np.sort(self.df_hotspot[col_date].dt.date.unique())
                        self.waktu_kordinat = pd.to_datetime(tanggal_unik)
                        
                        # Set batas maksimal slider sesuai jumlah hari unik yang ada apinya
                        self.slider_waktu.setMaximum(len(self.waktu_kordinat) - 1)
                    
                    total_api = len(self.df_hotspot)
                    
                    QMessageBox.information(self, "Success", f"Hotspot CSV loaded successfully!\nTotal unique days with hotspots: {len(self.waktu_kordinat)}\nTotal hotspot points: {total_api}")
                    
                # --- 2. JIKA FILE ADALAH GEOTIFF (.tif) ---
                elif file_path.endswith('.tif') or file_path.endswith('.tiff'):
                    # Baca GeoTIFF menggunakan rioxarray
                    ds = rioxarray.open_rasterio(file_path)
                    
                    try:
                        # gdf = gpd.read_file(path_shp_aktif)
                        # # GeoTIFF biasanya sudah punya CRS, kita langsung clip
                        # ds = ds.rio.clip(gdf.geometry, gdf.crs, drop=False)

                        gdf = gpd.read_file(path_shp_aktif)
                        # GeoTIFF biasanya sudah punya CRS, kita langsung clip
                        ds = ds.rio.clip(gdf.geometry, gdf.crs, drop=False)

                        # --- 🌟 TAMBAHAN: POTONG LAGI KHUSUS LAHAN GAMBUT ---
                        # Menggunakan drop=False sangat penting agar ukuran matriks (bingkai) tetap sama dengan ukuran pulau
                        gdf_gambut = gpd.read_file("shapefiles/Indonesia_peat_lands.shp")
                        gdf_gambut = gdf_gambut.to_crs("EPSG:4326")
                        ds = ds.rio.clip(gdf_gambut.geometry, gdf_gambut.crs, drop=False)
                        # ----------------------------------------------------

                        # Ambil batas asli dari data (minx, miny, maxx, maxy)
                        bounds = ds.rio.bounds()
                        extent_asli = [bounds[0], bounds[2], bounds[1], bounds[3]]

                    except Exception as e:
                        print(f"Error occurred while clipping GeoTIFF: {e}")
                    
                    # Ekstrak array nilai (Bentuknya: Band, Y, X)
                    array_data = ds.values
                    
                    # Karena GeoTIFF jarang punya metadata tanggal, kita buat tanggal otomatis
                    # Asumsi: jumlah band = jumlah hari, dimulai dari 1 Januari tahun ini
                    jumlah_hari = array_data.shape[0]
                    self.waktu_kordinat = pd.date_range(start='2024-01-01', periods=jumlah_hari, freq='D')
                    
                    self.slider_waktu.setMaximum(jumlah_hari - 1)
                    
                    # Filter nilai no-data khas CHIRPS (biasanya -9999) menjadi NaN
                    array_data = np.where(array_data < -100, np.nan, array_data)
                    self.data_hujan = array_data
                    self.extent_hujan = extent_asli
                    
                    QMessageBox.information(self, "Success", f"GeoTIFF loaded successfully!\nNumber of Bands/Days: {jumlah_hari}")

                # --- 3. JIKA FILE ADALAH NETCDF (.nc) ---
                elif file_path.endswith('.nc'):
                    ds = xr.open_dataset(file_path)
                    
                    # 1. POTONG SPASIAL DULU (Biar enteng prosesnya)
                    try:
                        gdf = gpd.read_file(path_shp_aktif)
                        # x_dim = 'longitude' if 'longitude' in ds.dims else 'lon'
                        # y_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
                        # ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                        # ds = ds.rio.write_crs("epsg:4326", inplace=True)
                        # ds = ds.rio.clip(gdf.geometry, gdf.crs, drop=False)

                        x_dim = 'longitude' if 'longitude' in ds.dims else 'lon'
                        y_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
                        ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                        ds = ds.rio.write_crs("epsg:4326", inplace=True)
                        ds = ds.rio.clip(gdf.geometry, gdf.crs, drop=False)

                        # --- 🌟 TAMBAHAN: POTONG LAGI KHUSUS LAHAN GAMBUT ---
                        gdf_gambut = gpd.read_file("shapefiles/Indonesia_peat_lands.shp")
                        gdf_gambut = gdf_gambut.to_crs("EPSG:4326")
                        ds = ds.rio.clip(gdf_gambut.geometry, gdf_gambut.crs, drop=False)
                        # ----------------------------------------------------
                    except Exception as e:
                        print(f"Warning Clipping NetCDF: {e}")
                    
                    # 2. CARI NAMA DIMENSI WAKTU
                    nama_waktu = 'valid_time' if 'valid_time' in ds.coords else 'time'
                    
                    # --- 3. KONVERSI JAM-JAMAN MENJADI HARIAN (RESAMPLING) ---
                    # Kita pisahkan logikanya berdasarkan variabel yang dipilih user
                    if "Temperature" in pilihan:
                        # Cari suhu TERTINGGI (max) dalam 1 hari ('1D')
                        ds = ds.resample({nama_waktu: '1D'}).max(dim=nama_waktu)
                        
                    elif "Rainfall" in pilihan:
                        # Jika data hujan Anda jam-jaman, harusnya DIJUMLAHKAN (sum) per hari
                        ds = ds.resample({nama_waktu: '1D'}).sum(dim=nama_waktu)
                        
                    elif "Soil Moisture" in pilihan:
                        # Kelembapan tanah biasanya diambil RATA-RATANYA (mean) per hari
                        ds = ds.resample({nama_waktu: '1D'}).mean(dim=nama_waktu)
                    # ---------------------------------------------------------
                    
                    # 4. EKSTRAK ARRAY DAN BATAS KOORDINAT (Seperti langkah sebelumnya)
                    nama_variabel = list(ds.data_vars)[0]
                    array_data = ds[nama_variabel].values
                    
                    bounds = ds.rio.bounds()
                    extent_asli = [bounds[0], bounds[2], bounds[1], bounds[3]]
                    
                    self.waktu_kordinat = pd.to_datetime(ds[nama_waktu].values)
                    self.slider_waktu.setMaximum(len(self.waktu_kordinat) - 1)
                    
                    # 5. SIMPAN KE VARIABEL KELAS
                    if "Rainfall" in pilihan:
                        self.data_hujan = array_data
                        self.extent_hujan = extent_asli
                    elif "Temperature" in pilihan:
                        # Konversi Kelvin ke Celcius jika diperlukan
                        if np.nanmean(array_data) > 200:
                            array_data = array_data - 273.15
                        self.data_suhu = array_data
                        self.extent_suhu = extent_asli
                    elif "Soil Moisture" in pilihan:
                        self.data_kelembapan = array_data
                        self.extent_kelembapan = extent_asli
                        
                    QMessageBox.information(self, "Success", f"Data NetCDF loaded!\nSuccessfully summarized into {len(self.waktu_kordinat)} days.")

                # REFRESH PETA
                self.update_map()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read file!\n{str(e)}")

    def update_map(self):
        pilihan = self.combo_variable.currentText()
        pilihan_region = self.combo_region.currentText()
        hari_idx = self.slider_waktu.value()
        
        # --- 1. PERBAIKAN: Bersihkan Seluruh Figure (Bingkai) ---
        self.canvas.fig.clear() 
        
        # Buat ulang kanvas utama yang ukurannya kembali normal 100%
        self.canvas.axes = self.canvas.fig.add_subplot(111)
        
        # Terapkan kembali tema terang karena fig.clear() mereset semuanya
        self.canvas.fig.patch.set_facecolor('#ffffff')
        self.canvas.axes.set_facecolor('#ffffff')
        self.canvas.axes.tick_params(colors='black')
        # ---------------------------------------------------------

        batas_aktif = self.config_region[pilihan_region]["extent"]
        shp_aktif = self.config_region[pilihan_region]["shp_batas"]
        
        # (Blok kode 'if hasattr(self, 'cbar')' yang lama HAPUS SAJA)

        tanggal_teks = f"Day {hari_idx + 1}"
        if hasattr(self, 'waktu_kordinat') and self.waktu_kordinat is not None:
            tanggal_teks = self.waktu_kordinat[hari_idx].strftime('%Y-%m-%d')
            self.label_hari_aktif.setText(tanggal_teks)
        else:
            self.label_hari_aktif.setText(tanggal_teks)

        judul = f"No data imported yet for {pilihan}"
        
        im = None 
        label_cbar = ""

        # --- 2. PLOT DATA ASLI DENGAN SKALA DINAMIS ---
        if "Rainfall" in pilihan and self.data_hujan is not None:
            cmap = 'Blues'
            judul = f"Rainfall Map ({tanggal_teks})"
            
            # AMBIL DATA HANYA UNTUK HARI INI
            data_hari_ini = self.data_hujan[hari_idx]
            
            # HITUNG VMIN/VMAX HANYA DARI DATA HARI INI
            vmin = np.nanmin(data_hari_ini)
            vmax = np.nanmax(data_hari_ini)
            
            # Penanganan jika seluruh peta isinya NaN (misal data kosong)
            if np.isnan(vmin) or np.isnan(vmax):
                 vmin, vmax = 0, 1 # Nilai default agar tidak error

            im = self.canvas.axes.imshow(data_hari_ini, cmap=cmap, extent=self.extent_hujan, vmin=vmin, vmax=vmax)
            label_cbar = "Rainfall (mm)"
            
        elif "Temperature" in pilihan and self.data_suhu is not None:
            cmap = 'YlOrRd'
            judul = f"Maximum Temperature Map ({tanggal_teks})"
            
            data_hari_ini = self.data_suhu[hari_idx]
            vmin = np.nanmin(data_hari_ini)
            vmax = np.nanmax(data_hari_ini)
            
            if np.isnan(vmin) or np.isnan(vmax):
                 vmin, vmax = 20, 35 # Rentang suhu default
                 
            im = self.canvas.axes.imshow(data_hari_ini, cmap=cmap, extent=self.extent_suhu, vmin=vmin, vmax=vmax)
            label_cbar = "Temperature (°C)"
            
        elif "Soil Moisture" in pilihan and self.data_kelembapan is not None:
            cmap = 'BrBG'
            judul = f"Soil Moisture Map ({tanggal_teks})"
            
            data_hari_ini = self.data_kelembapan[hari_idx]
            vmin = np.nanmin(data_hari_ini)
            vmax = np.nanmax(data_hari_ini)

            if np.isnan(vmin) or np.isnan(vmax):
                 vmin, vmax = 0, 1 # Rentang kelembapan default

            im = self.canvas.axes.imshow(data_hari_ini, cmap=cmap, extent=self.extent_kelembapan, vmin=vmin, vmax=vmax)
            label_cbar = "Soil Moisture (0 - 1)"
            
        elif "Hotspots" in pilihan:
            judul = f"Hotspots ({tanggal_teks})"
            #self.canvas.axes.imshow(np.zeros((100,100)), cmap='gray', alpha=0.3, extent=batas_sumatra, origin='lower')
            
            if self.df_hotspot is not None and self.waktu_kordinat is not None:
                tanggal_target = pd.to_datetime(tanggal_teks)
                col_date = 'acq_date' if 'acq_date' in self.df_hotspot.columns else 'date'
                df_hari_ini = self.df_hotspot[self.df_hotspot[col_date].dt.date == tanggal_target.date()]
                
                if not df_hari_ini.empty:
                    x_api = df_hari_ini['longitude'].values
                    y_api = df_hari_ini['latitude'].values
                    self.canvas.axes.scatter(x_api, y_api, color='red', marker='s', s=10, label='Hotspot')
                    self.canvas.axes.legend(loc='upper right')

        # --- 3. TAMBAHKAN COLORBAR BARU ---
        if im is not None:
            # Buat alat pemisah untuk kanvas utama
            divider = make_axes_locatable(self.canvas.axes)
            
            # Buat kotak baru di sebelah 'kanan' kanvas utama, selebar '5%' dari peta, jarak '0.1'
            cax = divider.append_axes("right", size="5%", pad=0.1)
            
            # Masukkan colorbar ke dalam kotak khusus (cax) tersebut
            self.cbar = self.canvas.fig.colorbar(im, cax=cax)
            self.cbar.set_label(label_cbar, color='black', fontweight='bold')
            self.cbar.ax.tick_params(colors='black')
            
            # Beri warna terang pada background kotak colorbar
            cax.set_facecolor('#ffffff')

        # --- KOSMETIK PETA ---
        try:
            gdf = gpd.read_file(shp_aktif)
            
            # HAPUS TANDA '#' DI DEPAN BARIS INI, DAN UBAH WARNANYA
            # facecolor='none' artinya tengahnya transparan
            # edgecolor='lightgray' agar garisnya terlihat di background gelap
            gdf.plot(ax=self.canvas.axes, facecolor='none', edgecolor='lightgray', linewidth=1.0)
            
        except Exception as e:
            # Tambahkan print error agar kita tahu jika file shp tidak ketemu
            print(f"Error loading shapefile: {e}")
            pass

        self.canvas.axes.set_title(judul, color='black', pad=10)
        self.canvas.axes.set_xlabel("Longitude", color='black')
        self.canvas.axes.set_ylabel("Latitude", color='black')
        self.canvas.axes.set_xlim([batas_aktif[0], batas_aktif[1]])
        self.canvas.axes.set_ylim([batas_aktif[2], batas_aktif[3]])
        
        # Kunci posisi peta secara absolut di tengah ('C' = Center)
        self.canvas.axes.set_anchor('C') 
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def mulai_training(self):
        """Fungsi yang dipanggil saat tombol Mulai ditekan"""
        
        # --- 1. VALIDASI (SATPAM) ---
        # Cek apakah semua 4 data sudah dimasukkan oleh user
        if self.data_hujan is None or self.data_suhu is None or self.data_kelembapan is None or self.df_hotspot is None:
            QMessageBox.warning(self, "Warning", "Please import all four datasets (Rainfall, Temperature, Soil Moisture, and Hotspots) in the Data Preparation tab first!")
            return
        
        # 1. Reset data history (list) agar kosong kembali
        # Sesuaikan dengan nama variabel list yang Anda gunakan
        self.history_loss = []
        self.history_val_loss = []
        self.history_epochs = []

        # 2. BERSIHKAN KANVAS PLOT
        self.ax_metrics.clear() # Menghapus semua garis lama
        
        # 3. SET ULANG LABEL (Karena .clear() menghapus semuanya termasuk teks)
        self.ax_metrics.set_xlabel('Epoch', color='black')
        self.ax_metrics.set_ylabel('Loss', color='black')
        
        # Gambar ulang kanvas kosongnya
        self.canvas_metrics.draw()

        # Ambil nilai dari UI
        epochs = self.spin_epochs.value()
        batch_size = self.spin_batch.value()
        time_steps = self.spin_timesteps.value()
        horizon = self.spin_horizon.value()
        layers = self.spin_layers.value()
        filters = self.spin_filters.value()
        dropout = self.spin_dropout.value()
        optimizer = self.combo_optimizer.currentText()
        loss_func = self.combo_loss.currentText()
        
        self.btn_train.setEnabled(False)
        QApplication.processEvents()
        
        # --- 2. RAKIT TENSORNYA ---
        try:
            extent = self.extent_suhu 
            
            # 1. SIAPKAN DATA MENTAH (BAHAN BAKU)
            # 1. SIAPKAN DATA MENTAH (BAHAN BAKU)
            hujan, suhu, kelem, hotspot = siapkan_data_mentah(
                self.data_hujan, self.data_suhu, self.data_kelembapan, 
                self.df_hotspot, self.waktu_kordinat, extent
            )
            
            total_hari = len(hujan)
            minimal_hari_dibutuhkan = (time_steps + 1) * 5 # Rumus aman untuk rasio 80:20
            
            # 2. CEK APAKAH DATA CUKUP BANYAK
            if total_hari < minimal_hari_dibutuhkan:
                QMessageBox.information(self, "Info Dataset", 
                                    f"Total data: {total_hari} hari.\n"
                                    f"Karena terlalu sedikit untuk dibagi dengan Time Steps {time_steps}, "
                                    "pelatihan akan menggunakan 100% data tanpa Validasi.")
                
                # Gunakan 100% data untuk training
                train_gen = FireDataGenerator(hujan, suhu, kelem, hotspot, time_steps, horizon, batch_size, shuffle=True)
                self.val_gen = None # Kosongkan validasi
                
            else:
                # BAGI DATA MENJADI TRAIN DAN VAL (80:20) SECARA NORMAL
                split_idx = int(total_hari * 0.8)
                hujan_tr, suhu_tr, kelem_tr, hotspot_tr = hujan[:split_idx], suhu[:split_idx], kelem[:split_idx], hotspot[:split_idx]
                hujan_val, suhu_val, kelem_val, hotspot_val = hujan[split_idx:], suhu[split_idx:], kelem[split_idx:], hotspot[split_idx:]
                
                train_gen = FireDataGenerator(hujan_tr, suhu_tr, kelem_tr, hotspot_tr, time_steps, horizon, batch_size, shuffle=True)
                self.val_gen = FireDataGenerator(hujan_val, suhu_val, kelem_val, hotspot_val, time_steps, horizon, batch_size, shuffle=False)
            
            # Ambil nilai threshold evaluasi dari UI
            eval_threshold = self.spin_eval_threshold.value()

            # 3. JALANKAN WORKER
            self.worker = TrainingWorker(
                epochs, batch_size, train_gen, self.val_gen,
                layers, filters, dropout, optimizer, loss_func,  # <-- Kirim param baru
                eval_threshold  # <-- Kirim nilai threshold evaluasi
            )
            
            self.worker.update_progress.connect(self.progress_bar.setValue)
            #self.worker.update_status.connect(self.label_status_ml.setText)
            self.worker.update_metrics.connect(self.update_grafik_training)
            self.worker.training_finished.connect(self.selesai_training)
            self.worker.sinyal_evaluasi.connect(self.tampilkan_hasil_evaluasi)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal: {str(e)}")
            self.selesai_training()

    def update_grafik_training(self, epoch, loss, val_loss):
        """Fungsi yang akan dipanggil setiap kali 1 epoch selesai"""
        self.history_epochs.append(epoch)
        self.history_loss.append(loss)
        self.history_val_loss.append(val_loss)
        
        self.ax_metrics.clear()
        self.ax_metrics.set_facecolor('#ffffff')
        self.ax_metrics.tick_params(colors='black')
        self.ax_metrics.set_xlabel('Epoch', color='black')
        self.ax_metrics.set_ylabel('Loss', color='black')
        
        # Gambar garis tren Loss (warna biru)
        # Gambar garis tren Loss (warna biru)
        self.ax_metrics.plot(self.history_epochs, self.history_loss, label='Train Loss', color='#0066cc', marker='o')
        
        # TAMBAHKAN BARIS INI: Gambar garis tren Validasi (warna merah)
        self.ax_metrics.plot(self.history_epochs, self.history_val_loss, label='Validation Loss', color='#cc0000', marker='s')

        self.ax_metrics.legend(loc='upper right')
        self.fig_metrics.tight_layout()
        self.canvas_metrics.draw()
    
    def simpan_model(self):
        """Membuka dialog dan menyimpan model yang ada di RAM ke Hard Disk"""
        
        # Keamanan ganda: Pastikan model benar-benar ada di memori
        if self.model_convlstm is None:
            QMessageBox.warning(self, "Error", "No trained model found! Please train a model first before saving.")
            return

        # Buka jendela dialog agar user bisa memilih lokasi dan nama file
        # Ekstensi default yang kita sarankan adalah .keras
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Trained Model", 
            "model_prediksi_gambut.keras", # Nama file rekomendasi
            "Keras Model (*.keras);;HDF5 Model (*.h5);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Proses menyimpan model
                self.btn_simpan_model.setEnabled(False)
                QApplication.processEvents() # Paksa UI update teks
                
                # Fungsi bawaan TensorFlow untuk menyimpan model utuh (Arsitektur + Bobot)
                self.model_convlstm.save(file_path)
                
                QMessageBox.information(self, "Success", f"Model successfully saved to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model:\n{str(e)}")
                
            finally:
                # Kembalikan tampilan tombol
                self.btn_simpan_model.setText("Save Model")
                self.btn_simpan_model.setEnabled(True)
    
    def jalankan_evaluasi_cepat(self):
        # Cek apakah model sudah pernah dilatih dan disimpan
        # 🌟 UBAH CEK FILE MENJADI CEK VARIABEL RAM
        if self.model_convlstm is None:
            QMessageBox.warning(self, "Error", "Model belum dilatih! Latih dulu di tab Build ML Model.")
            return
            
        if not hasattr(self, 'val_gen') or self.val_gen is None:
            QMessageBox.warning(self, "Error", "Data validasi tidak tersedia! Pastikan dataset cukup untuk split train/val.")
            return

        # Ambil threshold baru
        threshold_baru = self.spin_eval_threshold.value()
        
        # Ubah UI sementara
        self.btn_re_eval.setEnabled(False)
        # self.label_status_ml.setText("Memulai proses evaluasi ulang...")
        # self.label_val_precision.setText("Menghitung...")
        # self.label_val_recall.setText("Menghitung...")
        # self.label_val_f1.setText("Menghitung...")
        
        # Jalankan worker
        self.eval_worker = EvaluasiWorker(self.model_convlstm, self.val_gen, threshold_baru)
        
        # Hubungkan sinyal ke fungsi yang sudah ada
        self.eval_worker.sinyal_hasil.connect(self.tampilkan_hasil_evaluasi)
        # self.eval_worker.sinyal_status.connect(self.label_status_ml.setText)
        
        # Kembalikan tombol jika sudah selesai
        self.eval_worker.finished.connect(lambda: self.btn_re_eval.setEnabled(True))
        
        self.eval_worker.start()
    
    def tampilkan_hasil_evaluasi(self, precision, recall, f1, auc):
        """Fungsi untuk menangkap sinyal evaluasi dan mencetaknya ke GUI"""
        # Format angka menjadi 4 angka di belakang koma agar terlihat rapi
        self.label_val_precision.setText(f"{precision:.4f}")
        self.label_val_recall.setText(f"{recall:.4f}")
        self.label_val_f1.setText(f"{f1:.4f}")
        self.label_val_auc.setText(f"{auc:.4f}")

    def selesai_training(self):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("Start Training")
        self.btn_train.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 15px; border-radius: 5px;")
        
        # --- TAMBAHKAN KODE INI ---
        # Tangkap model dari worker jika pelatihannya berhasil (tidak error)
        if hasattr(self, 'worker') and hasattr(self.worker, 'model_hasil'):
            self.model_convlstm = self.worker.model_hasil

            self.btn_simpan_model.setEnabled(True)
        # -------------------------
    
    def muat_model(self):
        """Membuka dialog untuk memuat model dari penyimpanan lokal"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Trained Model File", 
            "", 
            "Keras Model (*.keras *.h5);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.btn_muat_model.setText("Loading Model...")
                QApplication.processEvents()

                self.model_convlstm = load_model(
                    file_path, 
                    custom_objects={
                        'loss': weighted_binary_crossentropy(1.0, 200.0),
                        'SliceSequence': SliceSequence 
                    },
                    compile=False,
                    safe_mode=False
                )

                # Cek atribut model untuk menyesuaikan UI 
                # Mencoba mendeteksi horizon dari layer Lambda jika memungkinkan
                try:
                    # Mencoba mengambil jumlah hari prediksi dari shape output model
                    # Output shape: (None, Horizon, Tinggi, Lebar, 1)
                    horizon_terdeteksi = self.model_convlstm.output_shape[1]
                    self.slider_hari.setMaximum(horizon_terdeteksi)
                except:
                    pass

                QMessageBox.information(self, "Success", f"Model loaded successfully!\nLocation: {file_path}")
                self.btn_prediksi.setEnabled(True) # Aktifkan tombol prediksi
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
                
            finally:
                self.btn_muat_model.setText("Load Pre-trained Model")
    
    def jalankan_prediksi(self):
        # 1. VALIDASI
        if self.data_hujan is None or self.data_suhu is None or self.data_kelembapan is None:
            QMessageBox.warning(self, "Error", "The data is not complete! Please make sure to import data for Rainfall, Temperature, and Humidity.")
            return
        
        if self.model_convlstm is None:
            QMessageBox.warning(self, "Error", "The model has not been built/trained yet! Please train the model first in the 'Build Model' tab.")
            return
            
        self.btn_prediksi.setEnabled(False)
        QApplication.processEvents()
        
        try:
            # 2. AMBIL DATA N HARI TERAKHIR
            time_steps = self.spin_timesteps.value() # Jika Anda sudah punya input ini
            
            # hujan_last = np.nan_to_num(self.data_hujan[-time_steps:], nan=0.0)
            # suhu_last = np.nan_to_num(self.data_suhu[-time_steps:], nan=0.0)
            # kelem_last = np.nan_to_num(self.data_kelembapan[-time_steps:], nan=0.0)

            # Hitung ekstremum keseluruhan dulu untuk mengisi area luar pulau
            max_hujan_global = np.nanmax(self.data_hujan) if np.any(~np.isnan(self.data_hujan)) else 0.0
            min_suhu_global = np.nanmin(self.data_suhu) if np.any(~np.isnan(self.data_suhu)) else 0.0

            # Ganti NaN sesuai aturan baru pada N hari terakhir
            hujan_last = np.nan_to_num(self.data_hujan[-time_steps:], nan=max_hujan_global)
            suhu_last = np.nan_to_num(self.data_suhu[-time_steps:], nan=min_suhu_global)
            kelem_last = np.nan_to_num(self.data_kelembapan[-time_steps:], nan=1.0)
            
            # 3. SAMAKAN UKURAN SPASIAL
            tinggi_target, lebar_target = hujan_last.shape[1], hujan_last.shape[2]
            
            def resize_last(arr):
                if arr.shape[1:] == (tinggi_target, lebar_target): return arr
                fy = tinggi_target / arr.shape[1]
                fx = lebar_target / arr.shape[2]
                return zoom(arr, (1.0, fy, fx), order=1)
                
            suhu_last = resize_last(suhu_last)
            kelem_last = resize_last(kelem_last)
            
            # Gabungkan jadi (1, time_steps, tinggi, lebar, 3)
            X_input = np.stack([hujan_last, suhu_last, kelem_last], axis=-1)
            X_input = np.expand_dims(X_input, axis=0) 
            
            # 4. LOAD MODEL DAN PREDIKSI!
            hasil_prediksi = self.model_convlstm.predict(X_input, verbose=0)
            
            # 🌟 PERUBAHAN MULTI-STEP: Ekstrak matriks 3D (Horizon, Tinggi, Lebar)
            # hasil_prediksi shape aslinya: (1, horizon, tinggi, lebar, 1)
            prediksi_3d = hasil_prediksi[0, :, :, :, 0]
            horizon = prediksi_3d.shape[0] # Ambil jumlah hari (horizon)
            
            extent = self.extent_suhu
            
            # Siapkan penampung kosong untuk hasil setelah di-masking (Peatland)
            masked_prediksi = np.zeros_like(prediksi_3d)

            # Lakukan masking peatland untuk setiap frame/hari
            try:
                path_gambut = r'shapefiles/Indonesia_peat_lands.shp' 
                gdf_gambut = gpd.read_file(path_gambut)
                
                # 🌟 PERBAIKAN DI SINI (Variabelnya ditukar agar sesuai sumbunya)
                # Longitude (X) = batas Kiri ke Kanan = ukuran LEBAR
                lons = np.linspace(extent[0], extent[1], lebar_target)
                
                # Latitude (Y) = batas Bawah ke Atas = ukuran TINGGI
                lats = np.linspace(extent[3], extent[2], tinggi_target)

                for i in range(horizon):
                    peta_2d = prediksi_3d[i]
                    da = xr.DataArray(peta_2d, coords={'y': lats, 'x': lons}, dims=('y', 'x'))
                    da = da.rio.write_crs("EPSG:4326")
                    da_masked = da.rio.clip(gdf_gambut.geometry.values, gdf_gambut.crs, drop=False)
                    masked_prediksi[i] = da_masked.values

            except Exception as e:
                print(f"Masking error. Displaying entire Sumatra. Error: {e}")
                masked_prediksi = prediksi_3d # Jika error, kembalikan ke bentuk asal
            
            # 🌟 5. SIMPAN HASIL KE MEMORI DAN NYALAKAN UI
            self.hasil_prediksi_sementara = masked_prediksi
            
            # Atur UI Slider Navigasi
            self.slider_hari.setMaximum(horizon)
            self.slider_hari.setEnabled(True)
            self.slider_hari.setValue(1) # Atur ke H+1
            
            # 6. PANGGIL FUNGSI GAMBAR (Untuk H+1)
            self.update_peta_prediksi(1)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Prediction", f"Failed to run prediction: {str(e)}")
            
        finally:
            self.btn_prediksi.setEnabled(True)
            self.btn_prediksi.setText("Fire Prediction")
    
    def tangani_hasil_prediksi(self, hasil_array):
        """Dipanggil saat Worker Prediksi selesai bekerja"""
        # hasil_array sekarang berbentuk (1, 3, H, W, 1)
        # Kita buang dimensi batch agar menjadi (3, H, W, 1)
        self.hasil_prediksi_sementara = hasil_array[0] 
        
        # Nyalakan slider
        self.slider_hari.setEnabled(True)
        self.slider_hari.setValue(1) # Reset ke hari pertama
        
        # Panggil fungsi gambar manual untuk hari pertama
        self.update_peta_prediksi(1)

    def update_peta_prediksi(self, hari_ke):
        # Cegah error jika fungsi dipanggil sebelum ada hasil prediksi
        if not hasattr(self, 'hasil_prediksi_sementara') or self.hasil_prediksi_sementara is None:
            return
        
        pilihan_region = self.combo_region.currentText()
        shp_aktif = self.config_region[pilihan_region]["shp_batas"]
        batas_aktif = self.config_region[pilihan_region]["extent"]

            
        # Ambil indeks (array mulai dari 0, H+1 berarti indeks 0)
        indeks = hari_ke - 1
        
        # Ambil peta untuk hari tersebut dari memori
        peta_probabilitas = self.hasil_prediksi_sementara[indeks].copy()
        
        # 1. TERAPKAN THRESHOLD
        batas = self.slider_threshold.value() 
        peta_probabilitas = np.where(peta_probabilitas < batas, np.nan, peta_probabilitas)
        
        # 2. GAMBAR KE KANVAS
        self.canvas_prediksi.fig.clear() 
        
        self.canvas_prediksi.axes = self.canvas_prediksi.fig.add_subplot(111)
        self.canvas_prediksi.axes.set_facecolor('#ffffff')
        self.canvas_prediksi.axes.tick_params(colors='black')
        
        extent = self.extent_suhu
        
        # Tumpuk dengan hasil prediksi
        im = self.canvas_prediksi.axes.imshow(peta_probabilitas, cmap='inferno', extent=extent, vmin=0, vmax=1, origin='upper')
        
        # Colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.canvas_prediksi.axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        self.cbar_prediksi = self.canvas_prediksi.fig.colorbar(im, cax=cax)
        self.cbar_prediksi.set_label("Fire Index", color='black', fontweight='bold')
        self.cbar_prediksi.ax.tick_params(colors='black')
        cax.set_facecolor('#ffffff')
        
        # Gambar Batas Sumatra
        try:
            gdf = gpd.read_file(shp_aktif)
            gdf.plot(ax=self.canvas_prediksi.axes, facecolor='none', edgecolor='lightgray', linewidth=1.0)
        except: pass
        
        # 3. KOSMETIK (Judul dinamis berdasarkan penambahan hari)
        import pandas as pd
        # Tambahkan jumlah hari_ke pada tanggal terakhir data input
        tanggal_prediksi = (self.waktu_kordinat[-1] + pd.Timedelta(days=hari_ke)).strftime('%Y-%m-%d')
        
        self.canvas_prediksi.axes.set_title(f"Fire Prediction ({tanggal_prediksi})", color='black', pad=15)
        self.canvas_prediksi.axes.set_xlabel("Longitude", color='black')
        self.canvas_prediksi.axes.set_ylabel("Latitude", color='black')
        self.canvas_prediksi.axes.set_xlim([batas_aktif[0], batas_aktif[1]])
        self.canvas_prediksi.axes.set_ylim([batas_aktif[2], batas_aktif[3]])
        self.canvas_prediksi.axes.set_anchor('C')
        
        self.canvas_prediksi.fig.tight_layout()
        self.canvas_prediksi.draw()
        
        # Update teks label di UI
        if hasattr(self, 'label_hari'):
            self.label_hari.setText(f"Menampilkan Prediksi: H+{hari_ke}")