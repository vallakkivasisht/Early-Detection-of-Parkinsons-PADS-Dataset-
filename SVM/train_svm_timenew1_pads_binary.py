"""
Train SVM on timenew1_preprocessed_pads_132ch .bin files
Binary classification: Healthy vs Parkinson's

The .bin files contain preprocessed sensor data with PADS-style features:
- 132 channels (8 tasks × 2 wrists × 8.25 features per axis, accounting for rotation)
- 976 timesteps
- Total size: 128832 float32 values per patient

This script:
1. Filters data to only include Healthy and Parkinson's patients
2. Excludes all other movement disorders
3. Extracts statistical features from the sequential data
4. Trains a binary SVM classifier
5. Reports accuracy, F1 score, precision, and recall
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("TRAINING BINARY SVM: HEALTHY vs PARKINSON'S")
print("Dataset: timenew1_preprocessed_pads_132ch + AGE AT DIAGNOSIS")
print("="*80)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/5] Loading preprocessed .bin files...")

DATA_DIR = 'timenew1_preprocessed_pads_132ch'
N_CHANNELS = 132
N_TIMESTEPS = 976

# Get list of available .bin files
bin_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_ml.bin')])
patient_ids = [int(f.split('_')[0]) for f in bin_files]

print(f"Found {len(bin_files)} preprocessed patients in {DATA_DIR}")
print(f"Patient IDs range: {min(patient_ids)} to {max(patient_ids)}")

# Load labels from patients_info_encoded.csv
print("\n[2/5] Loading labels and filtering for binary classification...")
labels_df = pd.read_csv('patients_info_encoded.csv')
labels_df['PatientID'] = labels_df['PatientID'].astype(int)

# Filter to only patients that have preprocessed data
labels_df = labels_df[labels_df['PatientID'].isin(patient_ids)]

print(f"\nOriginal class distribution (all patients):")
print(labels_df['condition'].value_counts())

# Filter for ONLY Healthy and Parkinson's patients
labels_df_binary = labels_df[labels_df['condition'].isin(['Healthy', "Parkinson's"])]

print(f"\nFiltered class distribution (Healthy vs Parkinson's only):")
print(labels_df_binary['condition'].value_counts())
print(f"\nTotal patients after filtering: {len(labels_df_binary)}")

# Load .bin files for filtered patients only
X_sequential = []
y = []
patient_ids_loaded = []
ages_at_diagnosis = []

print(f"\n[3/5] Loading data from .bin files for Healthy and Parkinson's patients...")
for _, row in labels_df_binary.iterrows():
    patient_id = row['PatientID']
    condition = row['condition']
    age_at_diagnosis = row['age_at_diagnosis']

    # Load .bin file
    bin_path = os.path.join(DATA_DIR, f'{patient_id:03d}_ml.bin')

    if not os.path.exists(bin_path):
        print(f"Warning: .bin file not found for patient {patient_id}, skipping...")
        continue

    data = np.fromfile(bin_path, dtype=np.float32)

    # Reshape to (132, 976)
    data_reshaped = data.reshape(N_CHANNELS, N_TIMESTEPS)

    X_sequential.append(data_reshaped)
    y.append(condition)
    patient_ids_loaded.append(patient_id)
    ages_at_diagnosis.append(age_at_diagnosis)

X_sequential = np.array(X_sequential)
y = np.array(y)
ages_at_diagnosis = np.array(ages_at_diagnosis)

print(f"Successfully loaded {len(X_sequential)} patients")
print(f"Data shape (sequential): {X_sequential.shape}")
print(f"Labels shape: {y.shape}")
print(f"Age at diagnosis stats: mean={np.nanmean(ages_at_diagnosis):.1f}, std={np.nanstd(ages_at_diagnosis):.1f}")

# Encode labels: Healthy=0, Parkinson's=1
y_encoded = np.array([1 if label == "Parkinson's" else 0 for label in y])

print(f"\nBinary encoding:")
print(f"  Healthy (0): {np.sum(y_encoded == 0)} samples")
print(f"  Parkinson's (1): {np.sum(y_encoded == 1)} samples")

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================
print(f"\n[4/5] Extracting advanced features from each channel...")

def bandpower(signal):
    """
    Compute bandpower features using FFT
    Returns 19 features corresponding to power in 1-19 Hz frequency bins
    """
    # Compute FFT
    fft_vals = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(len(signal))

    # Compute power spectrum
    power = np.abs(fft_vals) ** 2

    # Extract power in 1-19 Hz bins
    features = []
    for freq in range(1, 20):  # 1-19 Hz
        freq_mask = (fft_freq >= freq) & (fft_freq < freq + 1)
        band_power = np.sum(power[freq_mask])
        features.append(band_power)

    return np.array(features)

def std_windowed(signal):
    """
    Compute standard deviation in 4 time windows
    Returns 4 features
    """
    n = len(signal)
    window_size = n // 4
    features = []

    for i in range(4):
        start = i * window_size
        end = start + window_size if i < 3 else n
        window = signal[start:end]
        features.append(np.std(window))

    return np.array(features)

def abs_energy_windowed(signal):
    """
    Compute absolute energy (sum of squared values) in 4 time windows
    Returns 4 features
    """
    n = len(signal)
    window_size = n // 4
    features = []

    for i in range(4):
        start = i * window_size
        end = start + window_size if i < 3 else n
        window = signal[start:end]
        energy = np.sum(window ** 2)
        features.append(energy)

    return np.array(features)

def abs_max_windowed(signal):
    """
    Compute absolute maximum in 4 time windows
    Returns 4 features
    """
    n = len(signal)
    window_size = n // 4
    features = []

    for i in range(4):
        start = i * window_size
        end = start + window_size if i < 3 else n
        window = signal[start:end]
        features.append(np.max(np.abs(window)))

    return np.array(features)

# Define processing pipeline
processing_pipeline = [
    bandpower,           # 19 features (1-19 Hz frequency bins)
    std_windowed,        # 4 features (std in 4 time windows)
    abs_energy_windowed, # 4 features (energy in 4 windows)
    abs_max_windowed     # 4 features (max in 4 windows)
]
# Total: 31 features per channel

def feature_extraction(x):
    """
    Extract features from signal using the processing pipeline
    Applied to each channel independently

    Args:
        x: array of shape (n_channels, n_timesteps)

    Returns:
        features: array of shape (n_channels, 31) flattened to (n_channels * 31,)
    """
    features = []
    for func in processing_pipeline:
        features.append(np.apply_along_axis(func, 1, x))
    features = np.concatenate(features, axis=1)
    return features

def extract_features_from_sequential(X):
    """
    Extract features from all samples using the advanced feature extraction pipeline

    Args:
        X: array of shape (n_samples, n_channels, n_timesteps)

    Returns:
        features: array of shape (n_samples, n_channels * 31)
    """
    n_samples = X.shape[0]
    features_list = []

    for i in range(n_samples):
        # Extract features for this sample (shape: n_channels x n_timesteps)
        sample_features = feature_extraction(X[i])  # Returns (n_channels, 31)
        # Flatten to 1D array
        features_list.append(sample_features.flatten())

    return np.array(features_list)

X_features = extract_features_from_sequential(X_sequential)
print(f"Extracted sensor features shape: {X_features.shape}")
print(f"Total sensor features per sample: {X_features.shape[1]}")

# Add age at diagnosis as an additional feature
# Handle NaN values by replacing with median age
ages_cleaned = ages_at_diagnosis.copy()
median_age = np.nanmedian(ages_at_diagnosis)
ages_cleaned = np.where(np.isnan(ages_cleaned), median_age, ages_cleaned)
ages_cleaned = ages_cleaned.reshape(-1, 1)

# Concatenate age feature with sensor features
X_features_with_age = np.concatenate([X_features, ages_cleaned], axis=1)
print(f"Total features with age: {X_features_with_age.shape[1]} (sensor features + 1 age feature)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_features_with_age, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain/Test split:")
print(f"  Training samples: {len(X_train)}")
print(f"    - Healthy: {np.sum(y_train == 0)}")
print(f"    - Parkinson's: {np.sum(y_train == 1)}")
print(f"  Test samples: {len(X_test)}")
print(f"    - Healthy: {np.sum(y_test == 0)}")
print(f"    - Parkinson's: {np.sum(y_test == 1)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# TRAIN SVM
# ============================================================
print(f"\n[5/5] Training binary SVM classifier...")
print("-" * 80)

svm_model = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    class_weight='balanced',
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)
print("SVM training complete!")

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# ============================================================
# CALCULATE METRICS
# ============================================================
print("\n" + "="*80)
print("BINARY SVM MODEL PERFORMANCE: HEALTHY vs PARKINSON'S")
print("Features: Sensor Data (132 channels) + Age at Diagnosis")
print("="*80)

# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nOVERALL METRICS:")
print(f"  Accuracy:   {accuracy * 100:.2f}%")
print(f"  Precision:  {precision * 100:.2f}%")
print(f"  Recall:     {recall * 100:.2f}%")
print(f"  F1-Score:   {f1 * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nCONFUSION MATRIX:")
print(f"                  Predicted")
print(f"                  Healthy  Parkinson's")
print(f"Actual Healthy      {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"       Parkinson's  {cm[1,0]:3d}      {cm[1,1]:3d}")

# Classification report
print(f"\nDETAILED CLASSIFICATION REPORT:")
print("-" * 80)
target_names = ['Healthy', "Parkinson's"]
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# ============================================================
# SAVE RESULTS
# ============================================================
results_file = 'svm_timenew1_pads_binary_healthy_vs_parkinsons_with_age_results.txt'
with open(results_file, 'w') as f:
    f.write("BINARY SVM CLASSIFIER: HEALTHY vs PARKINSON'S\n")
    f.write("Dataset: timenew1_preprocessed_pads_132ch + Age at Diagnosis\n")
    f.write("="*80 + "\n\n")

    f.write(f"Dataset Information:\n")
    f.write(f"  Total patients (after filtering): {len(X_sequential)}\n")
    f.write(f"    - Healthy: {np.sum(y_encoded == 0)}\n")
    f.write(f"    - Parkinson's: {np.sum(y_encoded == 1)}\n")
    f.write(f"  Data shape: ({N_CHANNELS}, {N_TIMESTEPS})\n")
    f.write(f"  Feature extraction: Advanced pipeline + Age at diagnosis\n")
    f.write(f"  Total features: {X_features_with_age.shape[1]} (sensor: {X_features.shape[1]} + age: 1)\n")
    f.write(f"  Classification: Binary (Healthy=0, Parkinson's=1)\n\n")

    f.write("Excluded from training:\n")
    f.write("  - Other Movement Disorders\n")
    f.write("  - Atypical Parkinsonism\n")
    f.write("  - Essential Tremor\n")
    f.write("  - Multiple Sclerosis\n\n")

    f.write("="*80 + "\n")
    f.write("SVM MODEL CONFIGURATION\n")
    f.write("="*80 + "\n")
    f.write(f"  Kernel: rbf\n")
    f.write(f"  C: 10.0\n")
    f.write(f"  Gamma: scale\n")
    f.write(f"  Class weight: balanced\n\n")

    f.write("="*80 + "\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("="*80 + "\n\n")

    f.write(f"OVERALL METRICS:\n")
    f.write(f"  Accuracy:   {accuracy * 100:.2f}%\n")
    f.write(f"  Precision:  {precision * 100:.2f}%\n")
    f.write(f"  Recall:     {recall * 100:.2f}%\n")
    f.write(f"  F1-Score:   {f1 * 100:.2f}%\n\n")

    f.write("CONFUSION MATRIX:\n")
    f.write(f"                  Predicted\n")
    f.write(f"                  Healthy  Parkinson's\n")
    f.write(f"Actual Healthy      {cm[0,0]:3d}      {cm[0,1]:3d}\n")
    f.write(f"       Parkinson's  {cm[1,0]:3d}      {cm[1,1]:3d}\n\n")

    f.write("DETAILED CLASSIFICATION REPORT:\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

print(f"\n{'='*80}")
print(f"Results saved to: {results_file}")
print(f"{'='*80}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
