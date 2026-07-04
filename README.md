# ConvLSTM Hotspot Prediction

## Project Description

This repository contains the implementation of a deep learning model based on the **Convolutional Long Short-Term Memory (ConvLSTM)** architecture to predict the potential occurrence of forest and land fire hotspots. This modeling specifically handles spatio-temporal data to detect historical risks and geographical climate patterns, focusing on the Sumatra Island region and peatland covers in Indonesia.

The pipeline in this project facilitates spatial data preprocessing, input tensor dimension adjustments, and end-to-end model training. This spatio-temporal approach is highly ideal for integrating time-series environmental datasets—such as NASA FIRMS hotspot data and CHIRPS rainfall data—into geospatial grids.

## Repository Structure

```text
ConvLSTM-Hotspot-Prediction/
├── data/                       # Directory for storing historical spatial/temporal datasets
│   └── README.md
├── shapefiles/                 # Administrative boundaries and specific areas for spatial analysis
│   ├── batas_sumatra.*         # Vector files for the Sumatra Island boundary
│   └── Indonesia_peat_lands.*  # Vector files for Indonesia's peatland distribution
├── src/                        # Source code (GUI, Data Handler, ML Model, Workers)
│   ├── data_handler.py         
│   ├── gui.py                  
│   ├── ml_core.py              
│   └── workers.py              
├── .gitignore                  # Git ignore configuration file
├── main.py                     # Main script to run the program
├── README.md                   # Project documentation
└── requirements.txt            # List of Python dependencies

```

## Model Architecture

The neural network model in this repository is built purely using **ConvLSTM**. This architecture is designed to retain spatial characteristics from two-dimensional input data along the time (temporal) trajectory, enabling the model to comprehensively recognize the movement and expansion of hotspot areas. The architecture has been finalized on this approach to optimize computational efficiency and spatial accuracy.

## Prerequisites and Installation

Ensure Python is installed in your working environment. Clone this repository and install all listed dependencies using `pip`:

```bash
git clone https://github.com/username/ConvLSTM-Hotspot-Prediction.git
cd ConvLSTM-Hotspot-Prediction
pip install -r requirements.txt

```

*Note: Ensure spatial dependencies (such as `geopandas` or other QGIS support libraries) and `tensorflow` are properly configured in your environment.*

## Usage

1. **Data Preparation:** Place environmental datasets (hotspot data, rainfall, etc.) into the `data/` directory.
2. **Spatial Cropping (Optional):** Use the `.shp` files in the `shapefiles/` folder to filter or clip the main dataset to focus exclusively on the Sumatra region or peatland areas.
3. **Model Execution:** Run the main script to start the interface or processing pipeline (depending on `main.py` configuration):
```bash
python main.py

```

## Spatial Analysis

The `shapefiles` directory is provided for easy geographical mapping. These files can be visualized using QGIS or directly read within Python scripts using geospatial processing libraries to ensure the model's predictions are validated on accurate reference coordinates.
