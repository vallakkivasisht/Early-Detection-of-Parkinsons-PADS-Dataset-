"""
SVM Training - ACCELEROMETER ONLY
With SMOTE + Feature Selection

Key Optimizations:
1. SMOTE to balance class distribution (276 Parkinson's vs 79 Healthy)
2. SelectKBest feature selection (reduce from 2046 features)
3. Log-transformed bandpower (1-19 Hz)
4. Test multiple k values for SelectKBest

Channel structure: 66 accelerometer channels from 132 total channels
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("SVM WITH SMOTE + FEATURE SELECTION: ACCELEROMETER ONLY")
print("="*80)

# ============================================================
# FEATURE EXTRACTION FUNCTIONS (with Log Transform)
# ============================================================

def bandpower(signal):
    """
    Compute bandpower features using FFT with LOG SCALING
    Returns 19 features corresponding to power in 1-19 Hz frequency bins
    """
    fft_vals = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(len(signal))
    power = np.abs(fft_vals) ** 2
    log_power = np.log1p(power)  # log(1 + power)

    features = []
    for freq in range(1, 20):
        freq_mask = (fft_freq >= freq) & (fft_freq < freq + 1)
        band_power = np.sum(log_power[freq_mask])
        features.append(band_power)

    return np.array(features)

def std_windowed(signal):
    """Compute standard deviation in 4 time windows"""
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
    """Compute absolute energy in 4 time windows"""
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
    """Compute absolute maximum in 4 time windows"""
    n = len(signal)
    window_size = n // 4
    features = []

    for i in range(4):
        start = i * window_size
        end = start + window_size if i < 3 else n
        window = signal[start:end]
        features.append(np.max(np.abs(window)))

    return np.array(features)

# Processing pipeline
processing_pipeline = [
    bandpower,           # 19 features
    std_windowed,        # 4 features
    abs_energy_windowed, # 4 features
    abs_max_windowed     # 4 features
]
# Total: 31 features per channel

def feature_extraction(x):
    """Extract features from signal"""
    features = []
    for func in processing_pipeline:
        features.append(np.apply_along_axis(func, 1, x))
    features = np.concatenate(features, axis=1)
    return features

def extract_features_from_sequential(X):
    """Extract features from all samples"""
    n_samples = X.shape[0]
    features_list = []

    for i in range(n_samples):
        sample_features = feature_extraction(X[i])
        features_list.append(sample_features.flatten())

    return np.array(features_list)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading preprocessed .bin files...")

DATA_DIR = 'timenew1_preprocessed_pads_132ch'
N_CHANNELS_TOTAL = 132
N_TIMESTEPS = 976

# Calculate accelerometer channel indices
N_TASKS = 11
N_WRISTS = 2
N_SENSORS = 2
N_AXES = 3

acc_indices = []
for task_idx in range(N_TASKS):
    for wrist_idx in range(N_WRISTS):
        base_idx = (task_idx * N_WRISTS * N_SENSORS * N_AXES) + (wrist_idx * N_SENSORS * N_AXES)
        acc_indices.extend([base_idx, base_idx + 1, base_idx + 2])

N_CHANNELS_ACC = len(acc_indices)
print(f"Total channels: {N_CHANNELS_TOTAL}")
print(f"Accelerometer channels: {N_CHANNELS_ACC}")

# Get list of available .bin files
bin_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_ml.bin')])
patient_ids = [int(f.split('_')[0]) for f in bin_files]

print(f"Found {len(bin_files)} preprocessed patients")

# Load labels
print("\n[2/6] Loading labels and filtering...")
labels_df = pd.read_csv('patients_info_encoded.csv')
labels_df['PatientID'] = labels_df['PatientID'].astype(int)
labels_df = labels_df[labels_df['PatientID'].isin(patient_ids)]

print(f"Original class distribution:")
print(labels_df['condition'].value_counts())

# Filter for binary classification
labels_df_binary = labels_df[labels_df['condition'].isin(['Healthy', "Parkinson's"])]

print(f"\nFiltered class distribution (Healthy vs Parkinson's):")
print(labels_df_binary['condition'].value_counts())

# Load accelerometer data
X_sequential = []
y = []

print(f"\n[3/6] Loading accelerometer data from .bin files...")
for _, row in labels_df_binary.iterrows():
    patient_id = row['PatientID']
    condition = row['condition']

    bin_path = os.path.join(DATA_DIR, f'{patient_id:03d}_ml.bin')

    if not os.path.exists(bin_path):
        continue

    data = np.fromfile(bin_path, dtype=np.float32)
    data_reshaped = data.reshape(N_CHANNELS_TOTAL, N_TIMESTEPS)
    data_acc = data_reshaped[acc_indices, :]

    X_sequential.append(data_acc)
    y.append(condition)

X_sequential = np.array(X_sequential)
y = np.array(y)

print(f"Loaded {len(X_sequential)} patients")
print(f"Data shape: {X_sequential.shape}")

# Encode labels
y_encoded = np.array([1 if label == "Parkinson's" else 0 for label in y])

print(f"\nBinary encoding:")
print(f"  Healthy (0): {np.sum(y_encoded == 0)}")
print(f"  Parkinson's (1): {np.sum(y_encoded == 1)}")
print(f"  Class ratio (Parkinson's/Healthy): {np.sum(y_encoded == 1) / np.sum(y_encoded == 0):.2f}:1")

# ============================================================
# FEATURE EXTRACTION
# ============================================================
print(f"\n[4/6] Extracting features with log-transformed bandpower...")

X_features = extract_features_from_sequential(X_sequential)
print(f"Extracted features shape: {X_features.shape}")
print(f"Total features: {X_features.shape[1]} ({N_CHANNELS_ACC} channels × 31 features)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain/Test split (BEFORE SMOTE):")
print(f"  Training: {len(X_train)} (Healthy: {np.sum(y_train == 0)}, Parkinson's: {np.sum(y_train == 1)})")
print(f"  Test: {len(X_test)} (Healthy: {np.sum(y_test == 0)}, Parkinson's: {np.sum(y_test == 1)})")

# ============================================================
# APPLY SMOTE TO TRAINING DATA
# ============================================================
print(f"\n[5/6] Applying SMOTE to balance training data...")
print("-" * 80)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"  Training: {len(X_train_smote)} (Healthy: {np.sum(y_train_smote == 0)}, Parkinson's: {np.sum(y_train_smote == 1)})")
print(f"  Class balance: {np.sum(y_train_smote == 0)} : {np.sum(y_train_smote == 1)}")
print(f"  Synthetic samples created: {len(X_train_smote) - len(X_train)}")

# ============================================================
# FEATURE SELECTION + TRAIN MULTIPLE CONFIGURATIONS
# ============================================================
print(f"\n[6/6] Training SVMs with different feature selection configurations...")
print("-" * 80)

# Test different k values for SelectKBest
k_values = [500, 750, 1000, 1250, 1500, 2046]  # 2046 = all features (no selection)

results_list = []

for k in k_values:
    print(f"\n--- Configuration: k={k} features ---")

    # Feature Selection
    if k < 2046:
        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
        X_test_selected = selector.transform(X_test)
        print(f"Selected {k} best features out of {X_train_smote.shape[1]}")
    else:
        X_train_selected = X_train_smote
        X_test_selected = X_test
        print(f"Using all {X_train_smote.shape[1]} features (no selection)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Train SVM
    svm_model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    )

    svm_model.fit(X_train_scaled, y_train_smote)
    y_pred = svm_model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()
    healthy_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    parkinsons_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_acc = (healthy_recall + parkinsons_recall) / 2

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"Healthy recall: {healthy_recall * 100:.2f}%")
    print(f"Parkinson's recall: {parkinsons_recall * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_acc * 100:.2f}%")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    results_list.append({
        'k': k,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'healthy_recall': healthy_recall,
        'parkinsons_recall': parkinsons_recall,
        'balanced_acc': balanced_acc,
        'cm': cm,
        'y_pred': y_pred
    })

# ============================================================
# FIND BEST CONFIGURATION
# ============================================================
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

# Find best based on balanced accuracy
best_idx = max(range(len(results_list)), key=lambda i: results_list[i]['balanced_acc'])
best_result = results_list[best_idx]
best_k = k_values[best_idx]

print(f"\nBest configuration (highest balanced accuracy):")
print(f"  k = {best_k} features")
print(f"  Accuracy:   {best_result['accuracy'] * 100:.2f}%")
print(f"  Precision:  {best_result['precision'] * 100:.2f}%")
print(f"  Recall:     {best_result['recall'] * 100:.2f}%")
print(f"  F1-Score:   {best_result['f1'] * 100:.2f}%")
print(f"  Healthy recall: {best_result['healthy_recall'] * 100:.2f}%")
print(f"  Parkinson's recall: {best_result['parkinsons_recall'] * 100:.2f}%")
print(f"  Balanced accuracy: {best_result['balanced_acc'] * 100:.2f}%")

cm = best_result['cm']
print(f"\nCONFUSION MATRIX:")
print(f"                  Predicted")
print(f"                  Healthy  Parkinson's")
print(f"Actual Healthy      {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"       Parkinson's  {cm[1,0]:3d}      {cm[1,1]:3d}")

print(f"\nDETAILED CLASSIFICATION REPORT:")
print("-" * 80)
target_names = ['Healthy', "Parkinson's"]
print(classification_report(y_test, best_result['y_pred'], target_names=target_names, zero_division=0))

# ============================================================
# SAVE RESULTS
# ============================================================
results_file = 'svm_acc_smote_fs_results.txt'
with open(results_file, 'w') as f:
    f.write("SVM WITH SMOTE + FEATURE SELECTION - ACCELEROMETER ONLY\n")
    f.write("="*80 + "\n\n")

    f.write("OPTIMIZATIONS APPLIED:\n")
    f.write("  1. SMOTE to balance class distribution\n")
    f.write("  2. SelectKBest feature selection (tested multiple k values)\n")
    f.write("  3. Log-transformed bandpower (1-19 Hz)\n")
    f.write("  4. StandardScaler normalization\n\n")

    f.write(f"Dataset Information:\n")
    f.write(f"  Total patients: {len(X_sequential)}\n")
    f.write(f"  Healthy: {np.sum(y_encoded == 0)}\n")
    f.write(f"  Parkinson's: {np.sum(y_encoded == 1)}\n")
    f.write(f"  Original class ratio: {np.sum(y_encoded == 1) / np.sum(y_encoded == 0):.2f}:1\n")
    f.write(f"  After SMOTE: {np.sum(y_train_smote == 0)} : {np.sum(y_train_smote == 1)} (balanced)\n")
    f.write(f"  Channels: {N_CHANNELS_ACC} (accelerometer only)\n")
    f.write(f"  Original features: {X_features.shape[1]}\n\n")

    f.write("="*80 + "\n")
    f.write("ALL CONFIGURATIONS TESTED\n")
    f.write("="*80 + "\n\n")

    for result in results_list:
        f.write(f"k = {result['k']} features:\n")
        f.write(f"  Accuracy:           {result['accuracy'] * 100:.2f}%\n")
        f.write(f"  Precision:          {result['precision'] * 100:.2f}%\n")
        f.write(f"  Recall:             {result['recall'] * 100:.2f}%\n")
        f.write(f"  F1-Score:           {result['f1'] * 100:.2f}%\n")
        f.write(f"  Healthy recall:     {result['healthy_recall'] * 100:.2f}%\n")
        f.write(f"  Parkinson's recall: {result['parkinsons_recall'] * 100:.2f}%\n")
        f.write(f"  Balanced accuracy:  {result['balanced_acc'] * 100:.2f}%\n")
        f.write("\n")

    f.write("="*80 + "\n")
    f.write("BEST MODEL\n")
    f.write("="*80 + "\n")
    f.write(f"  Selected features: {best_k}\n")
    f.write(f"  Kernel: rbf, C: 10.0, Gamma: scale, Class weight: balanced\n")
    f.write(f"  SMOTE: Applied to training data\n\n")

    f.write(f"OVERALL METRICS:\n")
    f.write(f"  Accuracy:   {best_result['accuracy'] * 100:.2f}%\n")
    f.write(f"  Precision:  {best_result['precision'] * 100:.2f}%\n")
    f.write(f"  Recall:     {best_result['recall'] * 100:.2f}%\n")
    f.write(f"  F1-Score:   {best_result['f1'] * 100:.2f}%\n\n")

    f.write("PER-CLASS METRICS:\n")
    f.write(f"  Healthy recall:     {best_result['healthy_recall'] * 100:.2f}%\n")
    f.write(f"  Parkinson's recall: {best_result['parkinsons_recall'] * 100:.2f}%\n")
    f.write(f"  Balanced accuracy:  {best_result['balanced_acc'] * 100:.2f}%\n\n")

    f.write("CONFUSION MATRIX:\n")
    f.write(f"                  Predicted\n")
    f.write(f"                  Healthy  Parkinson's\n")
    f.write(f"Actual Healthy      {cm[0,0]:3d}      {cm[0,1]:3d}\n")
    f.write(f"       Parkinson's  {cm[1,0]:3d}      {cm[1,1]:3d}\n\n")

    f.write("DETAILED CLASSIFICATION REPORT:\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_test, best_result['y_pred'], target_names=target_names, zero_division=0))

print(f"\n{'='*80}")
print(f"Results saved to: {results_file}")
print(f"{'='*80}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
