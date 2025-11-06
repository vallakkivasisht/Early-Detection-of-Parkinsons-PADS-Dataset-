"""
Train CNN, ANN, and Random Forest models on PADS preprocessed data (132 channels)
Separate training for Accelerometer-only and Gyroscope-only data
Binary classification: Parkinson's vs Healthy with age at diagnosis

Uses the SAME feature extraction as other models:
- bandpower: 19 features (1-19 Hz frequency bins using FFT) - FREQUENCY FEATURES
- std_windowed: 4 features (std in 4 time windows)
- abs_energy_windowed: 4 features (sum of SQUARED values in 4 windows) - WINDOWED ENERGY
- abs_max_windowed: 4 features (max in 4 windows)
- age_at_diagnosis: 1 feature
Total: 31 features per channel + 1 age

Channel Breakdown:
- Total: 132 channels
- Accelerometer: 66 channels
- Gyroscope: 66 channels

Class Weights: Balanced (to handle class imbalance)
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("="*80)
print("TRAINING CNN, ANN, RF: ACCELEROMETER vs GYROSCOPE COMPARISON")
print("Dataset: PADS Preprocessed (132 channels) + Age at Diagnosis")
print("Classification: Parkinson's vs Healthy")
print("="*80)

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS (Same as other models)
# ============================================================================

def bandpower(signal):
    """
    Compute bandpower features using FFT
    Returns 19 features corresponding to power in 1-19 Hz frequency bins
    ** FREQUENCY-RELATED FEATURES **
    """
    fft_vals = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(len(signal))
    power = np.abs(fft_vals) ** 2

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
    Compute absolute energy (SUMMATION of SQUARED values) in 4 time windows
    ** WINDOWED ENERGY FEATURE **
    Returns 4 features
    """
    n = len(signal)
    window_size = n // 4
    features = []

    for i in range(4):
        start = i * window_size
        end = start + window_size if i < 3 else n
        window = signal[start:end]
        energy = np.sum(window ** 2)  # Sum of squared values
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
    abs_energy_windowed, # 4 features (SUM OF SQUARED values in 4 windows)
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
        features: array of shape (n_channels, 31)
    """
    features = []
    for func in processing_pipeline:
        features.append(np.apply_along_axis(func, 1, x))
    features = np.concatenate(features, axis=1)
    return features

def extract_features_from_sequential(X):
    """
    Extract features from all samples

    Args:
        X: array of shape (n_samples, n_channels, n_timesteps)

    Returns:
        features: array of shape (n_samples, n_channels * 31)
    """
    n_samples = X.shape[0]
    features_list = []

    for i in range(n_samples):
        if (i + 1) % 50 == 0:
            print(f"  Extracting features: {i+1}/{n_samples}")
        sample_features = feature_extraction(X[i])  # Returns (n_channels, 31)
        features_list.append(sample_features.flatten())

    return np.array(features_list)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.3)

        self.fc5 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_length):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.4)

        # Calculate size after convolutions
        self.flat_size = 256 * (input_length // 8)

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)

        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)

        x = self.fc3(x)
        return x

# ============================================================================
# IDENTIFY ACCELEROMETER AND GYROSCOPE CHANNELS
# ============================================================================

print("\nIdentifying accelerometer and gyroscope channels...")

# Load channel names
channel_names_file = 'timenew1_preprocessed_pads_132ch/channel_names.txt'
with open(channel_names_file, 'r') as f:
    channel_names = [line.strip() for line in f.readlines() if line.strip()]

# Identify indices
acc_indices = [i for i, name in enumerate(channel_names) if 'Accelerometer' in name]
gyro_indices = [i for i, name in enumerate(channel_names) if 'Gyroscope' in name]

print(f"Total channels: {len(channel_names)}")
print(f"Accelerometer channels: {len(acc_indices)}")
print(f"Gyroscope channels: {len(gyro_indices)}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading patient information...")
patients_df = pd.read_csv('patients_info_encoded.csv')

# Filter for Parkinson's and Healthy only
print("Filtering for Parkinson's and Healthy patients...")
patients_df = patients_df[(patients_df['COND_PARKINSONS'] == 1) | (patients_df['COND_HEALTHY'] == 1)].copy()
print(f"Total patients after filtering: {len(patients_df)}")
print(f"Parkinson's patients: {patients_df['COND_PARKINSONS'].sum()}")
print(f"Healthy patients: {patients_df['COND_HEALTHY'].sum()}")

# Handle missing age_at_diagnosis
patients_df['age_at_diagnosis'] = patients_df['age_at_diagnosis'].fillna(patients_df['age'])
median_age = patients_df['age_at_diagnosis'].median()
patients_df['age_at_diagnosis'] = patients_df['age_at_diagnosis'].fillna(median_age)

# Load preprocessed data
print("\nLoading preprocessed PADS data...")
data_dir = 'timenew1_preprocessed_pads_132ch'
N_CHANNELS = 132
N_TIMESTEPS = 976

X_sequential = []
y_list = []
ages_list = []

for idx, row in patients_df.iterrows():
    patient_id = row['PatientID']
    file_path = os.path.join(data_dir, f'{patient_id:03d}_ml.bin')

    if os.path.exists(file_path):
        data = np.fromfile(file_path, dtype=np.float32)
        data = data.reshape(N_CHANNELS, N_TIMESTEPS)  # (132 channels, 976 timesteps)

        X_sequential.append(data)
        y_list.append(row['COND_PARKINSONS'])  # 1 for Parkinson's, 0 for Healthy
        ages_list.append(row['age_at_diagnosis'])
    else:
        print(f"Warning: File not found for patient {patient_id}")

X_sequential = np.array(X_sequential)
y_array = np.array(y_list)
ages_array = np.array(ages_list).reshape(-1, 1)

print(f"Loaded data for {len(X_sequential)} patients")
print(f"X shape (sequential): {X_sequential.shape}")

# Separate accelerometer and gyroscope data
X_acc = X_sequential[:, acc_indices, :]  # (n_patients, 66, 976)
X_gyro = X_sequential[:, gyro_indices, :]  # (n_patients, 66, 976)

print(f"\nAccelerometer data shape: {X_acc.shape}")
print(f"Gyroscope data shape: {X_gyro.shape}")

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_array), y=y_array)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights (balanced): {class_weights_dict}")

# ============================================================================
# FUNCTION TO TRAIN AND EVALUATE MODELS
# ============================================================================

def train_evaluate_models(X_data, y_data, ages, data_type, class_weights_dict):
    """
    Train and evaluate Random Forest, ANN, and CNN models

    Returns:
        Dictionary containing results for all three models
    """
    print("\n" + "="*80)
    print(f"TRAINING MODELS - {data_type}")
    print("="*80)

    n_channels = X_data.shape[1]

    # Extract features
    print(f"\nExtracting features from {n_channels} channels...")
    X_features = extract_features_from_sequential(X_data)
    print(f"Extracted features shape: {X_features.shape}")

    # Add age
    X_features_with_age = np.concatenate([X_features, ages], axis=1)
    print(f"Features with age: {X_features_with_age.shape[1]} (sensor: {X_features.shape[1]} + age: 1)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features_with_age, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    print(f"\nTrain/Test split:")
    print(f"  Training: {len(X_train)} samples (Healthy: {np.sum(y_train == 0)}, Parkinson's: {np.sum(y_train == 1)})")
    print(f"  Test: {len(X_test)} samples (Healthy: {np.sum(y_test == 0)}, Parkinson's: {np.sum(y_test == 1)})")

    results = {}

    # ========================================================================
    # RANDOM FOREST
    # ========================================================================
    print("\n" + "-"*80)
    print("TRAINING RANDOM FOREST")
    print("-"*80)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    results['rf'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf, zero_division=0),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
        'y_pred': y_pred_rf
    }

    print(f"Random Forest - Accuracy: {results['rf']['accuracy']:.4f}, F1: {results['rf']['f1']:.4f}")

    # ========================================================================
    # ANN
    # ========================================================================
    print("\n" + "-"*80)
    print("TRAINING ANN")
    print("-"*80)

    # Standardize for ANN
    scaler_ann = StandardScaler()
    X_train_ann = scaler_ann.fit_transform(X_train)
    X_test_ann = scaler_ann.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_ann).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_ann).to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    input_size = X_train_ann.shape[1]
    ann_model = ANNModel(input_size).to(device)

    # Loss and optimizer with class weights
    class_weights_tensor = torch.FloatTensor([class_weights_dict[0], class_weights_dict[1]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(ann_model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training
    num_epochs = 50
    for epoch in range(num_epochs):
        ann_model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = ann_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    ann_model.eval()
    with torch.no_grad():
        outputs = ann_model(X_test_tensor)
        _, y_pred_ann = torch.max(outputs, 1)
        y_pred_ann = y_pred_ann.cpu().numpy()

    results['ann'] = {
        'accuracy': accuracy_score(y_test, y_pred_ann),
        'f1': f1_score(y_test, y_pred_ann, zero_division=0),
        'precision': precision_score(y_test, y_pred_ann, zero_division=0),
        'recall': recall_score(y_test, y_pred_ann, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_ann),
        'y_pred': y_pred_ann
    }

    print(f"ANN - Accuracy: {results['ann']['accuracy']:.4f}, F1: {results['ann']['f1']:.4f}")

    # ========================================================================
    # CNN
    # ========================================================================
    print("\n" + "-"*80)
    print("TRAINING CNN")
    print("-"*80)

    # Prepare data for CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], 1, -1)
    X_test_cnn = X_test.reshape(X_test.shape[0], 1, -1)

    # Normalize
    X_train_cnn = (X_train_cnn - X_train_cnn.mean()) / (X_train_cnn.std() + 1e-8)
    X_test_cnn = (X_test_cnn - X_test_cnn.mean()) / (X_test_cnn.std() + 1e-8)

    # Convert to tensors
    X_train_cnn_tensor = torch.FloatTensor(X_train_cnn).to(device)
    y_train_cnn_tensor = torch.LongTensor(y_train).to(device)
    X_test_cnn_tensor = torch.FloatTensor(X_test_cnn).to(device)

    # Create DataLoader
    train_dataset_cnn = TensorDataset(X_train_cnn_tensor, y_train_cnn_tensor)
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)

    # Initialize model
    input_length = X_train_cnn.shape[2]
    cnn_model = CNNModel(input_length).to(device)

    # Loss and optimizer
    criterion_cnn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training
    for epoch in range(num_epochs):
        cnn_model.train()
        for batch_X, batch_y in train_loader_cnn:
            optimizer_cnn.zero_grad()
            outputs = cnn_model(batch_X)
            loss = criterion_cnn(outputs, batch_y)
            loss.backward()
            optimizer_cnn.step()

    # Evaluation
    cnn_model.eval()
    with torch.no_grad():
        outputs = cnn_model(X_test_cnn_tensor)
        _, y_pred_cnn = torch.max(outputs, 1)
        y_pred_cnn = y_pred_cnn.cpu().numpy()

    results['cnn'] = {
        'accuracy': accuracy_score(y_test, y_pred_cnn),
        'f1': f1_score(y_test, y_pred_cnn, zero_division=0),
        'precision': precision_score(y_test, y_pred_cnn, zero_division=0),
        'recall': recall_score(y_test, y_pred_cnn, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_cnn),
        'y_pred': y_pred_cnn
    }

    print(f"CNN - Accuracy: {results['cnn']['accuracy']:.4f}, F1: {results['cnn']['f1']:.4f}")

    # Print summary
    print("\n" + "="*80)
    print(f"SUMMARY - {data_type}")
    print("="*80)
    print(f"Random Forest: Accuracy={results['rf']['accuracy']:.4f}, F1={results['rf']['f1']:.4f}")
    print(f"ANN:           Accuracy={results['ann']['accuracy']:.4f}, F1={results['ann']['f1']:.4f}")
    print(f"CNN:           Accuracy={results['cnn']['accuracy']:.4f}, F1={results['cnn']['f1']:.4f}")

    return {
        'data_type': data_type,
        'n_channels': n_channels,
        'n_features': X_features_with_age.shape[1],
        'y_test': y_test,
        'results': results
    }

# ============================================================================
# TRAIN MODELS FOR EACH DATA TYPE
# ============================================================================

print("\n" + "="*80)
print("TRAINING ALL MODELS ON DIFFERENT SENSOR COMBINATIONS")
print("="*80)

# Train on Accelerometer only
results_acc = train_evaluate_models(X_acc, y_array, ages_array, "ACCELEROMETER ONLY", class_weights_dict)

# Train on Gyroscope only
results_gyro = train_evaluate_models(X_gyro, y_array, ages_array, "GYROSCOPE ONLY", class_weights_dict)

# Train on Combined
results_combined = train_evaluate_models(X_sequential, y_array, ages_array, "COMBINED (ACC + GYRO)", class_weights_dict)

# ============================================================================
# CREATE COMPARISON
# ============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON - ALL MODELS AND SENSORS")
print("="*80)

comparison_data = []
for result_set, label in [(results_acc, 'Acc'), (results_gyro, 'Gyro'), (results_combined, 'Combined')]:
    for model_name in ['rf', 'ann', 'cnn']:
        model_label = {'rf': 'Random Forest', 'ann': 'ANN', 'cnn': 'CNN'}[model_name]
        res = result_set['results'][model_name]
        comparison_data.append({
            'Model': f"{model_label} ({label})",
            'Sensor': label,
            'Channels': result_set['n_channels'],
            'Features': result_set['n_features'],
            'Accuracy': res['accuracy'],
            'F1 Score': res['f1'],
            'Precision': res['precision'],
            'Recall': res['recall']
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv('cnn_ann_rf_pads_acc_gyro_comparison_results.csv', index=False)
print("\n\nResults saved to: cnn_ann_rf_pads_acc_gyro_comparison_results.csv")

# Save detailed results
with open('cnn_ann_rf_pads_acc_gyro_comparison_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CNN, ANN, RF COMPARISON: ACCELEROMETER vs GYROSCOPE vs COMBINED\n")
    f.write("Dataset: PADS Preprocessed + Age at Diagnosis\n")
    f.write("Classification: Parkinson's vs Healthy\n")
    f.write("="*80 + "\n\n")

    f.write("Feature Extraction (Same for all):\n")
    f.write("  - bandpower: 19 features (1-19 Hz frequency bins) ** FREQUENCY FEATURES **\n")
    f.write("  - std_windowed: 4 features (std in 4 time windows)\n")
    f.write("  - abs_energy_windowed: 4 features (SUM OF SQUARED values) ** WINDOWED ENERGY **\n")
    f.write("  - abs_max_windowed: 4 features (max in 4 windows)\n")
    f.write("  - age_at_diagnosis: 1 feature\n")
    f.write("  Total: 31 features per channel + 1 age\n\n")

    f.write(f"Dataset Information:\n")
    f.write(f"  Total patients: {len(X_sequential)}\n")
    f.write(f"  Parkinson's: {np.sum(y_array == 1)}\n")
    f.write(f"  Healthy: {np.sum(y_array == 0)}\n")
    f.write(f"  Class weights: {class_weights_dict}\n\n")

    f.write(f"Channel Breakdown:\n")
    f.write(f"  Total channels: {N_CHANNELS}\n")
    f.write(f"  Accelerometer channels: {len(acc_indices)}\n")
    f.write(f"  Gyroscope channels: {len(gyro_indices)}\n\n")

    f.write("="*80 + "\n")
    f.write("COMPARISON SUMMARY (Sorted by Accuracy)\n")
    f.write("="*80 + "\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")

    # Detailed results for each configuration
    for result_set in [results_acc, results_gyro, results_combined]:
        f.write("="*80 + "\n")
        f.write(f"{result_set['data_type']} - DETAILED RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Channels: {result_set['n_channels']}\n")
        f.write(f"Total features: {result_set['n_features']}\n\n")

        for model_name, model_label in [('rf', 'Random Forest'), ('ann', 'ANN'), ('cnn', 'CNN')]:
            res = result_set['results'][model_name]
            f.write(f"\n{model_label}:\n")
            f.write(f"  Accuracy:  {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)\n")
            f.write(f"  F1 Score:  {res['f1']:.4f}\n")
            f.write(f"  Precision: {res['precision']:.4f}\n")
            f.write(f"  Recall:    {res['recall']:.4f}\n")
            f.write("\n  Classification Report:\n")
            f.write(classification_report(result_set['y_test'], res['y_pred'],
                                        target_names=['Healthy', 'Parkinsons'], zero_division=0))

print("Detailed results saved to: cnn_ann_rf_pads_acc_gyro_comparison_results.txt")

# Create visualization
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Accuracy comparison
ax1 = fig.add_subplot(gs[0, :])
models = comparison_df['Model']
accuracies = comparison_df['Accuracy']
colors = ['#2ecc71' if 'Acc' in m else '#3498db' if 'Gyro' in m else '#e74c3c' for m in models]
bars = ax1.barh(range(len(models)), accuracies, color=colors)
ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models)
ax1.set_xlabel('Accuracy')
ax1.set_title('Model Accuracy Comparison (All Sensors)')
ax1.set_xlim([0, 1])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontsize=8)

# Confusion matrices for best models
# Get best model for each sensor
best_acc = max(results_acc['results'].items(), key=lambda x: x[1]['accuracy'])
best_gyro = max(results_gyro['results'].items(), key=lambda x: x[1]['accuracy'])
best_combined = max(results_combined['results'].items(), key=lambda x: x[1]['accuracy'])

# Plot confusion matrices
for idx, (result_set, best, title) in enumerate([
    (results_acc, best_acc, 'Accelerometer'),
    (results_gyro, best_gyro, 'Gyroscope'),
    (results_combined, best_combined, 'Combined')
]):
    ax = fig.add_subplot(gs[1, idx])
    cm = best[1]['confusion_matrix']
    model_name = {'rf': 'RF', 'ann': 'ANN', 'cnn': 'CNN'}[best[0]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['Healthy', 'Parkinsons'],
               yticklabels=['Healthy', 'Parkinsons'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'{title} - Best: {model_name}\nAcc: {best[1]["accuracy"]:.4f}')

plt.savefig('cnn_ann_rf_pads_acc_gyro_comparison_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: cnn_ann_rf_pads_acc_gyro_comparison_results.png")

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
