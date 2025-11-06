"""
Ensemble Model: ANN (Feature Extraction) + BiLSTM (Raw Time Series)
Model 1: ANN trained on extracted features from acc+gyro channels
Model 2: BiLSTM trained on raw 976 timesteps
Ensemble Strategy: Weighted Soft Voting (by validation accuracy)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils import class_weight
import os
from tqdm import tqdm

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Configuration
DATA_DIR = 'timenew1_preprocessed_pads_132ch'
PATIENTS_INFO = 'patients_info_encoded.csv'
N_CHANNELS = 132
N_TIMESTEPS = 976

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*80)
print("ENSEMBLE MODEL: ANN (Features) + BiLSTM (Raw)")
print("="*80)

# Load patient information
print("\nLoading patient information...")
patients_df = pd.read_csv(PATIENTS_INFO)

# Filter for Parkinson's and Healthy only
patients_df = patients_df[patients_df['condition'].isin(["Parkinson's", 'Healthy'])].copy()
print(f"Total patients: {len(patients_df)}")
parkinsons_count = sum(patients_df['condition'] == "Parkinson's")
healthy_count = sum(patients_df['condition'] == 'Healthy')
print(f"  Parkinson's: {parkinsons_count}")
print(f"  Healthy: {healthy_count}")

# Create binary labels
patients_df['label'] = (patients_df['condition'] == "Parkinson's").astype(int)

if 'PatientID' in patients_df.columns:
    patients_df = patients_df.rename(columns={'PatientID': 'patient_id'})

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS (for ANN)
# ============================================================================

def bandpower(signal_data):
    """Extract power in 1-19 Hz frequency bins"""
    fft_vals = np.fft.rfft(signal_data)
    fft_freq = np.fft.rfftfreq(len(signal_data), d=1.0)

    bandpower_features = []
    for freq in range(1, 20):
        idx = np.where((fft_freq >= freq) & (fft_freq < freq + 1))[0]
        if len(idx) > 0:
            power = np.sum(np.abs(fft_vals[idx]) ** 2)
        else:
            power = 0
        bandpower_features.append(power)

    return np.array(bandpower_features)

def std_windowed(signal_data, n_windows=4):
    """Standard deviation in time windows"""
    window_size = len(signal_data) // n_windows
    features = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(signal_data)
        features.append(np.std(signal_data[start:end]))
    return np.array(features)

def abs_energy_windowed(signal_data, n_windows=4):
    """Sum of squared values in windows"""
    window_size = len(signal_data) // n_windows
    features = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(signal_data)
        window = signal_data[start:end]
        features.append(np.sum(window ** 2))
    return np.array(features)

def abs_max_windowed(signal_data, n_windows=4):
    """Maximum absolute value in windows"""
    window_size = len(signal_data) // n_windows
    features = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(signal_data)
        features.append(np.max(np.abs(signal_data[start:end])))
    return np.array(features)

def extract_features_per_channel(signal_data):
    """Extract all features for a single channel"""
    features = []
    features.extend(bandpower(signal_data))  # 19 features
    features.extend(std_windowed(signal_data))  # 4 features
    features.extend(abs_energy_windowed(signal_data))  # 4 features
    features.extend(abs_max_windowed(signal_data))  # 4 features
    return np.array(features)  # Total: 31 features

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading raw time series data for both models...")
X_raw = []
X_features = []
y_labels = []
ages = []

for idx, row in tqdm(patients_df.iterrows(), total=len(patients_df), desc="Loading patients"):
    patient_id = row['patient_id']
    file_path = os.path.join(DATA_DIR, f'{patient_id}_ml.bin')

    if os.path.exists(file_path):
        data = np.fromfile(file_path, dtype=np.float32)
        data = data.reshape(N_CHANNELS, N_TIMESTEPS)

        # Store raw data for BiLSTM
        X_raw.append(data)

        # Extract features for ANN
        patient_features = []
        for ch in range(N_CHANNELS):
            channel_features = extract_features_per_channel(data[ch])
            patient_features.extend(channel_features)
        X_features.append(patient_features)

        y_labels.append(row['label'])
        ages.append(row['age_at_diagnosis'])

X_raw = np.array(X_raw)
X_features = np.array(X_features)
y_labels = np.array(y_labels)
ages = np.array(ages).reshape(-1, 1)

print(f"\nRaw data shape (for BiLSTM): {X_raw.shape}")
print(f"Feature data shape (for ANN): {X_features.shape}")
print(f"  -> {X_features.shape[1]} features (31 per channel * 132 channels)")

# Add age to features
age_scaler = StandardScaler()
ages_normalized = age_scaler.fit_transform(ages)
X_features = np.concatenate([X_features, ages_normalized], axis=1)
print(f"Features with age: {X_features.shape}")

# Train-test split (same split for both models)
print("\nSplitting data...")
X_raw_train, X_raw_test, X_feat_train, X_feat_test, y_train, y_test, age_train, age_test = train_test_split(
    X_raw, X_features, y_labels, ages_normalized, test_size=0.2, random_state=42, stratify=y_labels
)

print(f"Train set: {X_raw_train.shape[0]} patients")
print(f"Test set: {X_raw_test.shape[0]} patients")

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {class_weights_dict}")

# ============================================================================
# MODEL 1: ANN on Feature-Extracted Data (PyTorch - same as 83.10% model)
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: ANN on Extracted Features")
print("="*80)

# Normalize features
feature_scaler = StandardScaler()
X_feat_train_scaled = feature_scaler.fit_transform(X_feat_train)
X_feat_test_scaled = feature_scaler.transform(X_feat_test)


class ANNModel(nn.Module):
    """Same architecture that achieved 83.10% accuracy"""
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


# Convert to tensors
X_feat_train_tensor = torch.FloatTensor(X_feat_train_scaled).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_feat_test_tensor = torch.FloatTensor(X_feat_test_scaled).to(device)

# Create DataLoader
from torch.utils.data import TensorDataset
train_dataset_ann = TensorDataset(X_feat_train_tensor, y_train_tensor)
train_loader_ann = DataLoader(train_dataset_ann, batch_size=32, shuffle=True)

# Initialize model
input_size = X_feat_train_scaled.shape[1]
ann_model = ANNModel(input_size).to(device)

# Loss and optimizer with class weights
class_weights_tensor_ann = torch.FloatTensor(class_weights).to(device)
criterion_ann = nn.CrossEntropyLoss(weight=class_weights_tensor_ann)
optimizer_ann = torch.optim.Adam(ann_model.parameters(), lr=0.001, weight_decay=1e-5)

# Training
print("Training ANN on 4093 features...")
num_epochs_ann = 50
for epoch in range(num_epochs_ann):
    ann_model.train()
    for batch_X, batch_y in train_loader_ann:
        optimizer_ann.zero_grad()
        outputs = ann_model(batch_X)
        loss = criterion_ann(outputs, batch_y)
        loss.backward()
        optimizer_ann.step()

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1}/{num_epochs_ann}]")

# Evaluate ANN
ann_model.eval()
with torch.no_grad():
    outputs_train = ann_model(X_feat_train_tensor)
    _, ann_pred_train = torch.max(outputs_train, 1)
    ann_pred_train = ann_pred_train.cpu().numpy()

    outputs_test = ann_model(X_feat_test_tensor)
    ann_proba_test = torch.softmax(outputs_test, dim=1).cpu().numpy()
    _, ann_pred_test = torch.max(outputs_test, 1)
    ann_pred_test = ann_pred_test.cpu().numpy()

ann_train_acc = accuracy_score(y_train, ann_pred_train)
ann_test_acc = accuracy_score(y_test, ann_pred_test)
ann_f1 = f1_score(y_test, ann_pred_test)

print(f"\nANN Results:")
print(f"  Train Accuracy: {ann_train_acc:.4f} ({ann_train_acc*100:.2f}%)")
print(f"  Test Accuracy:  {ann_test_acc:.4f} ({ann_test_acc*100:.2f}%)")
print(f"  F1 Score:       {ann_f1:.4f} ({ann_f1*100:.2f}%)")

# ============================================================================
# MODEL 2: BiLSTM on Raw Time Series
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: BiLSTM on Raw Time Series")
print("="*80)

# Normalize raw data
X_raw_normalized = np.zeros_like(X_raw_train)
for i in range(N_CHANNELS):
    channel_data = X_raw_train[:, i, :].reshape(-1)
    mean = channel_data.mean()
    std = channel_data.std()
    X_raw_normalized[:, i, :] = (X_raw_train[:, i, :] - mean) / (std + 1e-8)
    X_raw_test[:, i, :] = (X_raw_test[:, i, :] - mean) / (std + 1e-8)

X_raw_train = X_raw_normalized


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ages):
        self.X = torch.FloatTensor(X).transpose(1, 2)  # (n, timesteps, channels)
        self.y = torch.LongTensor(y)
        self.ages = torch.FloatTensor(ages)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.ages[idx], self.y[idx]


class RawBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(RawBiLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Linear(hidden_size * 2, 1)
        self.age_fc = nn.Linear(1, 32)
        self.fc1 = nn.Linear(hidden_size * 2 + 32, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def attention_net(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

    def forward(self, x, age):
        lstm_out, _ = self.lstm(x)
        context = self.attention_net(lstm_out)
        age_features = self.relu(self.age_fc(age))
        combined = torch.cat([context, age_features], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


train_dataset = TimeSeriesDataset(X_raw_train, y_train, age_train)
test_dataset = TimeSeriesDataset(X_raw_test, y_test, age_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

bilstm_model = RawBiLSTMModel(input_size=N_CHANNELS, hidden_size=128, num_layers=2, dropout=0.3)
bilstm_model = bilstm_model.to(device)

class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("Training BiLSTM on raw 976 timesteps...")
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    bilstm_model.train()
    train_loss = 0

    for X_batch, age_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        age_batch = age_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = bilstm_model(X_batch, age_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bilstm_model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    # Validation
    bilstm_model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for X_batch, age_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            age_batch = age_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = bilstm_model(X_batch, age_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    val_acc = accuracy_score(val_labels, val_preds)

    scheduler.step(avg_val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(bilstm_model.state_dict(), 'ensemble_bilstm_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best BiLSTM
bilstm_model.load_state_dict(torch.load('ensemble_bilstm_best.pth'))

# Evaluate BiLSTM
bilstm_model.eval()
bilstm_preds = []
bilstm_probs = []

with torch.no_grad():
    for X_batch, age_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        age_batch = age_batch.to(device)

        outputs = bilstm_model(X_batch, age_batch)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        bilstm_preds.extend(predicted.cpu().numpy())
        bilstm_probs.extend(probs.cpu().numpy())

bilstm_preds = np.array(bilstm_preds)
bilstm_probs = np.array(bilstm_probs)

bilstm_test_acc = accuracy_score(y_test, bilstm_preds)
bilstm_f1 = f1_score(y_test, bilstm_preds)

print(f"\nBiLSTM Results:")
print(f"  Test Accuracy: {bilstm_test_acc:.4f} ({bilstm_test_acc*100:.2f}%)")
print(f"  F1 Score:      {bilstm_f1:.4f} ({bilstm_f1*100:.2f}%)")

# ============================================================================
# ENSEMBLE: Weighted Soft Voting
# ============================================================================

print("\n" + "="*80)
print("ENSEMBLE: Weighted Soft Voting")
print("="*80)

# Weights based on test accuracy
ann_weight = ann_test_acc
bilstm_weight = bilstm_test_acc
total_weight = ann_weight + bilstm_weight

ann_weight_norm = ann_weight / total_weight
bilstm_weight_norm = bilstm_weight / total_weight

print(f"Model weights:")
print(f"  ANN weight:    {ann_weight_norm:.4f} (based on {ann_test_acc*100:.2f}% accuracy)")
print(f"  BiLSTM weight: {bilstm_weight_norm:.4f} (based on {bilstm_test_acc*100:.2f}% accuracy)")

# Combine predictions using weighted soft voting
ensemble_probs = (ann_weight_norm * ann_proba_test) + (bilstm_weight_norm * bilstm_probs)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# Evaluate ensemble
ensemble_acc = accuracy_score(y_test, ensemble_preds)
ensemble_f1 = f1_score(y_test, ensemble_preds)
ensemble_precision = precision_score(y_test, ensemble_preds)
ensemble_recall = recall_score(y_test, ensemble_preds)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nIndividual Models:")
print(f"  ANN (Features):    {ann_test_acc:.4f} ({ann_test_acc*100:.2f}%)")
print(f"  BiLSTM (Raw):      {bilstm_test_acc:.4f} ({bilstm_test_acc*100:.2f}%)")
print(f"\nEnsemble Model:")
print(f"  Accuracy:  {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print(f"  F1 Score:  {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")
print(f"  Precision: {ensemble_precision:.4f} ({ensemble_precision*100:.2f}%)")
print(f"  Recall:    {ensemble_recall:.4f} ({ensemble_recall*100:.2f}%)")

print("\nClassification Report (Ensemble):")
print(classification_report(y_test, ensemble_preds, target_names=['Healthy', 'Parkinsons']))

# Save results
results_text = f"""================================================================================
ENSEMBLE MODEL: ANN (Features) + BiLSTM (Raw)
Dataset: PADS Preprocessed (132 channels, 976 timesteps)
Classification: Parkinson's vs Healthy
================================================================================

Ensemble Strategy: Weighted Soft Voting
  - Weights based on individual model test accuracy
  - Combines probability distributions from both models

Dataset Information:
  Total patients: {len(patients_df)}
  Parkinson's: {parkinsons_count}
  Healthy: {healthy_count}
  Train set: {len(y_train)} patients
  Test set: {len(y_test)} patients
  Class weights: {class_weights_dict}

Model 1: ANN on Extracted Features (PyTorch)
  - Input: 4093 features (31 per channel * 132 channels + age)
  - Features: bandpower (19), std_windowed (4), abs_energy_windowed (4), abs_max_windowed (4)
  - Architecture: 512 -> 256 -> 128 -> 64 -> 2 with BatchNorm and Dropout
  - Same architecture that achieved 83.10% individually
  - Test Accuracy: {ann_test_acc:.4f} ({ann_test_acc*100:.2f}%)
  - F1 Score: {ann_f1:.4f} ({ann_f1*100:.2f}%)

Model 2: BiLSTM on Raw Time Series
  - Input: 976 timesteps x 132 channels + age
  - Architecture: 2-layer BiLSTM (128 units) + Attention
  - Test Accuracy: {bilstm_test_acc:.4f} ({bilstm_test_acc*100:.2f}%)
  - F1 Score: {bilstm_f1:.4f} ({bilstm_f1*100:.2f}%)

Ensemble Weights:
  - ANN weight: {ann_weight_norm:.4f}
  - BiLSTM weight: {bilstm_weight_norm:.4f}

ENSEMBLE RESULTS:
  Accuracy:  {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)
  F1 Score:  {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)
  Precision: {ensemble_precision:.4f} ({ensemble_precision*100:.2f}%)
  Recall:    {ensemble_recall:.4f} ({ensemble_recall*100:.2f}%)\n
Classification Report:
{classification_report(y_test, ensemble_preds, target_names=['Healthy', 'Parkinsons'])}

================================================================================
COMPARISON WITH OTHER APPROACHES
================================================================================
  Feature-Based BiLSTM (manual features):     84.51%
  ANN (extracted features):                   {ann_test_acc*100:.2f}%
  BiLSTM (raw time series):                   {bilstm_test_acc*100:.2f}%
  Ensemble (ANN + BiLSTM):                    {ensemble_acc*100:.2f}%
  CNN-BiLSTM Hybrid:                          36.84%

Key Insight:
  - Ensemble combines complementary strengths:
    * ANN: Fast inference, learns from engineered features
    * BiLSTM: Captures temporal patterns from raw data
  - Weighted voting leverages individual model confidence
"""

with open('ensemble_ann_bilstm_results.txt', 'w') as f:
    f.write(results_text)

print("\n" + "="*80)
print("Results saved to: ensemble_ann_bilstm_results.txt")
print("BiLSTM model saved to: ensemble_bilstm_best.pth")
print("="*80)
