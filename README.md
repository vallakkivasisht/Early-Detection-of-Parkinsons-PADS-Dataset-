#  Early Detection of Parkinson’s Disease using Wearable Sensors

A comprehensive machine learning pipeline for **early detection of Parkinson’s Disease (PD)** using **accelerometer and gyroscope data** from the **PADS (Parkinson’s Disease Smartwatch)** dataset.  
The project achieves up to **84.5% accuracy** with SVM and includes implementations of **six different ML/DL models**.

---

## Features

-  **End-to-End ML Pipeline** — Tests both accelerometer and gyroscope data to predict Parkinson’s disease.  
-  **Multi-Model Comparison** — Evaluated on multiple models including ANN, CNN, BiLSTM, SVM, Random Forest, and an Ensemble (ANN + BiLSTM).  
-  **Rigorous Feature Engineering** — Extensive preprocessing (L1 trend filtering, artifact removal) and extraction of key motion features such as RMS, energy, frequency, and variability.  
-  **Versatile Prediction** — Can accurately predict Parkinson’s across various ML models using both raw time-series and statistical features.  

---

---

##  Dataset Specifications

- **469 patients** total  
  - Parkinson’s Disease: 298 patients (63.5%)  
  - Healthy Controls: 96 patients (20.5%)  
  - Other Movement Disorders: 75 patients (16.0%)  
- **132 channels** of preprocessed accelerometer + gyroscope data  
- **L1 trend filtering (λ = 50)** to remove gravity drift  
- **11 motor tasks** capturing various movement types  
- **High-quality smartwatch recordings** with artifact removal  

---

##  Quick Start

### Using Python Virtual Environment (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   git clone https://github.com/vallakkivasisht/Early-Detection-of-Parkinsons-PADS-Dataset-.git
   cd parkinsons-detection/working_model
   
## Architecture Overview
### Preprocessing Pipeline

- **L1 Trend Filtering (λ=50)**: Removes gravity and drift
- **Artifact Removal**: Trims initial smartwatch vibration samples
- **Channel Standardization**: 132 total (11 tasks × 2 sensors × 3 axes)
- **Patient-Level Splitting**: Prevents data leakage (80/20 train/test)

### Traditional ML Models

- **Random Forest**: 200 trees, max_depth=10, balanced class weights
- **SVM**: RBF kernel, tuned with GridSearchCV
- **Ensemble**: Weighted voting combination of models

### Deep Learning Models

- **BiLSTM**: 2 bidirectional layers (128 hidden units) for temporal learning
- **CNN-1D**: 4 convolutional layers for spatial feature extraction
- **ANN**: 4 dense layers using engineered statistical features

### Data Management

- **Preprocessed Data**: Stored as binary .bin files (float32)
- **Metadata**: CSV containing demographics and diagnosis labels
- **Extracted Features**: ~388 per patient (statistical + frequency domain)

| Category              | Tools/Libraries                                          |
| --------------------- | -------------------------------------------------------- |
| **ML/DL Frameworks**  | scikit-learn, TensorFlow, PyTorch                        |
| **Data Processing**   | NumPy, Pandas, SciPy                                     |
| **Signal Processing** | CVXPY (L1 trend filtering)                               |
| **Visualization**     | Matplotlib, Seaborn                                      |
| **Model Storage**     | Joblib (`.pkl`), PyTorch (`.pth`), TensorFlow (`.keras`) |

### Installation Requirements

- **numpy**>=1.21.0
- **pandas**>=1.3.0
- **scikit-learn**>=1.0.0
- **tensorflow**>=2.8.0
- **torch**>=1.10.0
- **matplotlib**>=3.4.0
- **seaborn**>=0.11.0
- **cvxpy**>=1.2.0
- **joblib**>=1.1.0
- **scipy**>=1.7.0

