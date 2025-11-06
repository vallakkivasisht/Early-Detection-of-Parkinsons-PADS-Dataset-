"""
Train CNN, ANN, and Random Forest models on PADS preprocessed data
with Parkinson's vs Healthy classification, including age at diagnosis.

Uses the SAME feature extraction and class weights as the SVM model for fair comparison.

Feature Extraction Pipeline:
- bandpower: 19 features (power in 1-19 Hz frequency bins using FFT)
- std_windowed: 4 features (std in 4 time windows)
- abs_energy_windowed: 4 features (energy in 4 windows)
- abs_max_windowed: 4 features (max in 4 windows)
Total: 31 features per channel × 132 channels = 4092 features + 1 age = 4093 total

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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS (Same as SVM)
# ============================================================================

def bandpower(signal):
    """
    Compute bandpower features using FFT
    Returns 19 features corresponding to power in 1-19 Hz frequency bins

    ** FREQUENCY-RELATED FEATURES **
    This function extracts frequency-domain information by:
    1. Computing FFT (Fast Fourier Transform) to convert time-domain signal to frequency domain
    2. Calculating power spectrum (squared magnitude of FFT)
    3. Extracting power in each 1 Hz bin from 1-19 Hz

    These features capture tremor frequencies and other frequency characteristics
    important for Parkinson's detection.
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

# Define processing pipeline (SAME AS SVM)
processing_pipeline = [
    bandpower,           # 19 features (1-19 Hz frequency bins) ** FREQUENCY FEATURES **
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
        features: array of shape (n_channels, 31)
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
        if (i + 1) % 50 == 0:
            print(f"  Extracting features: {i+1}/{n_samples}")
        # Extract features for this sample (shape: n_channels x n_timesteps)
        sample_features = feature_extraction(X[i])  # Returns (n_channels, 31)
        # Flatten to 1D array
        features_list.append(sample_features.flatten())

    return np.array(features_list)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading patient information...")
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
        # Load binary data
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
print(f"y shape: {y_array.shape}")
print(f"ages shape: {ages_array.shape}")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

print("\n" + "="*80)
print("FEATURE EXTRACTION (Same as SVM)")
print("="*80)
print("\nExtracting features using pipeline:")
print("  1. bandpower: 19 features (1-19 Hz frequency bins) ** FREQUENCY FEATURES **")
print("  2. std_windowed: 4 features (std in 4 time windows)")
print("  3. abs_energy_windowed: 4 features (energy in 4 windows)")
print("  4. abs_max_windowed: 4 features (max in 4 windows)")
print(f"  Total: 31 features × {N_CHANNELS} channels = {31 * N_CHANNELS} features")

X_features = extract_features_from_sequential(X_sequential)
print(f"\nExtracted sensor features shape: {X_features.shape}")

# Add age as a feature
X_features_with_age = np.concatenate([X_features, ages_array], axis=1)
print(f"Total features with age: {X_features_with_age.shape[1]} (sensor: {X_features.shape[1]} + age: 1)")

# Split data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_features_with_age, y_array, test_size=0.2, random_state=42, stratify=y_array
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"  Healthy: {np.sum(y_train == 0)}")
print(f"  Parkinson's: {np.sum(y_train == 1)}")
print(f"Test set: {X_test.shape[0]} samples")
print(f"  Healthy: {np.sum(y_test == 0)}")
print(f"  Parkinson's: {np.sum(y_test == 1)}")

# Calculate class weights (SAME AS SVM: balanced)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights (balanced): {class_weights_dict}")

# ============================================================================
# 1. RANDOM FOREST (with class weights)
# ============================================================================
print("\n" + "="*80)
print("TRAINING RANDOM FOREST (with balanced class weights)")
print("="*80)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # SAME AS SVM
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Healthy', 'Parkinsons']))

# ============================================================================
# 2. ANN (with class weights)
# ============================================================================
print("\n" + "="*80)
print("TRAINING ANN (with weighted loss)")
print("="*80)

# Standardize
scaler_ann = StandardScaler()
X_train_ann = scaler_ann.fit_transform(X_train)
X_test_ann = scaler_ann.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_ann).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test_ann).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define ANN model
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

# Initialize model
input_size = X_train_ann.shape[1]
ann_model = ANNModel(input_size).to(device)

# Create weighted loss (SAME AS SVM: balanced)
class_weights_tensor = torch.FloatTensor([class_weights[0], class_weights[1]]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(ann_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training
print(f"ANN input size: {input_size}")
print(f"Using class weights in loss: {class_weights_tensor.cpu().numpy()}")
print("Training ANN...")
num_epochs = 50

for epoch in range(num_epochs):
    ann_model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = ann_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step(train_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

# Evaluation
ann_model.eval()
with torch.no_grad():
    outputs = ann_model(X_test_tensor)
    _, y_pred_ann = torch.max(outputs, 1)
    y_pred_ann = y_pred_ann.cpu().numpy()

ann_accuracy = accuracy_score(y_test, y_pred_ann)
ann_f1 = f1_score(y_test, y_pred_ann)
ann_precision = precision_score(y_test, y_pred_ann)
ann_recall = recall_score(y_test, y_pred_ann)

print("\nANN Results:")
print(f"Accuracy:  {ann_accuracy:.4f}")
print(f"F1 Score:  {ann_f1:.4f}")
print(f"Precision: {ann_precision:.4f}")
print(f"Recall:    {ann_recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ann, target_names=['Healthy', 'Parkinsons']))

# ============================================================================
# 3. CNN (with class weights)
# ============================================================================
print("\n" + "="*80)
print("TRAINING CNN (with weighted loss)")
print("="*80)

# For CNN, we'll use the extracted features reshaped as "channels"
# Reshape features: (n_samples, n_features) -> (n_samples, 1, n_features)
X_train_cnn = X_train.reshape(X_train.shape[0], 1, -1)
X_test_cnn = X_test.reshape(X_test.shape[0], 1, -1)

# Normalize
X_train_cnn = (X_train_cnn - X_train_cnn.mean()) / (X_train_cnn.std() + 1e-8)
X_test_cnn = (X_test_cnn - X_test_cnn.mean()) / (X_test_cnn.std() + 1e-8)

# Convert to tensors
X_train_cnn_tensor = torch.FloatTensor(X_train_cnn).to(device)
X_test_cnn_tensor = torch.FloatTensor(X_test_cnn).to(device)

# Create DataLoader
train_dataset_cnn = TensorDataset(X_train_cnn_tensor, y_train_tensor)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)

# Define CNN model
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

# Initialize model
input_length = X_train_cnn.shape[2]
cnn_model = CNNModel(input_length).to(device)

print(f"CNN input shape: (1 channel, {input_length} features)")

# Loss and optimizer (with class weights)
criterion_cnn = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.5, patience=5)

print(f"Using class weights in loss: {class_weights_tensor.cpu().numpy()}")
print("Training CNN...")
num_epochs = 50

for epoch in range(num_epochs):
    cnn_model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader_cnn:
        optimizer_cnn.zero_grad()
        outputs = cnn_model(batch_X)
        loss = criterion_cnn(outputs, batch_y)
        loss.backward()
        optimizer_cnn.step()
        train_loss += loss.item()

    train_loss /= len(train_loader_cnn)
    scheduler_cnn.step(train_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

# Evaluation
cnn_model.eval()
with torch.no_grad():
    outputs = cnn_model(X_test_cnn_tensor)
    _, y_pred_cnn = torch.max(outputs, 1)
    y_pred_cnn = y_pred_cnn.cpu().numpy()

cnn_accuracy = accuracy_score(y_test, y_pred_cnn)
cnn_f1 = f1_score(y_test, y_pred_cnn)
cnn_precision = precision_score(y_test, y_pred_cnn)
cnn_recall = recall_score(y_test, y_pred_cnn)

print("\nCNN Results:")
print(f"Accuracy:  {cnn_accuracy:.4f}")
print(f"F1 Score:  {cnn_f1:.4f}")
print(f"Precision: {cnn_precision:.4f}")
print(f"Recall:    {cnn_recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cnn, target_names=['Healthy', 'Parkinsons']))

# ============================================================================
# SUMMARY AND COMPARISON
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - MODEL COMPARISON (with Balanced Class Weights)")
print("="*80)

results_df = pd.DataFrame({
    'Model': ['Random Forest', 'ANN', 'CNN'],
    'Accuracy': [rf_accuracy, ann_accuracy, cnn_accuracy],
    'F1 Score': [rf_f1, ann_f1, cnn_f1],
    'Precision': [rf_precision, ann_precision, cnn_precision],
    'Recall': [rf_recall, ann_recall, cnn_recall]
})

print(results_df.to_string(index=False))

# Save results
results_df.to_csv('cnn_ann_rf_pads_with_age_balanced_results.csv', index=False)
print("\nResults saved to: cnn_ann_rf_pads_with_age_balanced_results.csv")

# Save detailed results to text file
with open('cnn_ann_rf_pads_with_age_balanced_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CNN, ANN, and Random Forest Training Results (with Balanced Class Weights)\n")
    f.write("Dataset: PADS Preprocessed Features + Age at Diagnosis\n")
    f.write("Classification: Parkinson's vs Healthy\n")
    f.write("="*80 + "\n\n")

    f.write("Feature Extraction (Same as SVM):\n")
    f.write("  - bandpower: 19 features (1-19 Hz frequency bins) ** FREQUENCY FEATURES **\n")
    f.write("  - std_windowed: 4 features (std in 4 time windows)\n")
    f.write("  - abs_energy_windowed: 4 features (energy in 4 windows)\n")
    f.write("  - abs_max_windowed: 4 features (max in 4 windows)\n")
    f.write(f"  Total: 31 features × {N_CHANNELS} channels = {31 * N_CHANNELS} features + 1 age\n\n")

    f.write(f"Total patients: {len(X_sequential)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Total features: {X_features_with_age.shape[1]}\n")
    f.write(f"Class weights (balanced): {class_weights_dict}\n\n")

    f.write("="*80 + "\n")
    f.write("RANDOM FOREST RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Accuracy:  {rf_accuracy:.4f}\n")
    f.write(f"F1 Score:  {rf_f1:.4f}\n")
    f.write(f"Precision: {rf_precision:.4f}\n")
    f.write(f"Recall:    {rf_recall:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=['Healthy', 'Parkinsons']))
    f.write("\n")

    f.write("="*80 + "\n")
    f.write("ANN RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Accuracy:  {ann_accuracy:.4f}\n")
    f.write(f"F1 Score:  {ann_f1:.4f}\n")
    f.write(f"Precision: {ann_precision:.4f}\n")
    f.write(f"Recall:    {ann_recall:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_ann, target_names=['Healthy', 'Parkinsons']))
    f.write("\n")

    f.write("="*80 + "\n")
    f.write("CNN RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Accuracy:  {cnn_accuracy:.4f}\n")
    f.write(f"F1 Score:  {cnn_f1:.4f}\n")
    f.write(f"Precision: {cnn_precision:.4f}\n")
    f.write(f"Recall:    {cnn_recall:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_cnn, target_names=['Healthy', 'Parkinsons']))
    f.write("\n")

    f.write("="*80 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("="*80 + "\n")
    f.write(results_df.to_string(index=False))

print("Detailed results saved to: cnn_ann_rf_pads_with_age_balanced_results.txt")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Accuracy comparison
ax1 = axes[0, 0]
models = results_df['Model']
accuracies = results_df['Accuracy']
bars1 = ax1.bar(models, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison (Balanced Weights)')
ax1.set_ylim([0, 1])
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom')

# Plot 2: All metrics comparison
ax2 = axes[0, 1]
x = np.arange(len(models))
width = 0.2
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics):
    values = results_df[metric]
    ax2.bar(x + i*width, values, width, label=metric, color=colors[i])

ax2.set_ylabel('Score')
ax2.set_title('All Metrics Comparison (Balanced Weights)')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim([0, 1])

# Plot 3: Confusion Matrix - CNN
ax3 = axes[1, 0]
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=ax3,
           xticklabels=['Healthy', 'Parkinsons'],
           yticklabels=['Healthy', 'Parkinsons'])
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')
ax3.set_title('CNN Confusion Matrix')

# Plot 4: Confusion Matrix - Random Forest
ax4 = axes[1, 1]
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax4,
           xticklabels=['Healthy', 'Parkinsons'],
           yticklabels=['Healthy', 'Parkinsons'])
ax4.set_ylabel('True Label')
ax4.set_xlabel('Predicted Label')
ax4.set_title('Random Forest Confusion Matrix')

plt.tight_layout()
plt.savefig('cnn_ann_rf_pads_with_age_balanced_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: cnn_ann_rf_pads_with_age_balanced_results.png")

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
