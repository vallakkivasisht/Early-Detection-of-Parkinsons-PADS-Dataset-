"""
Pure BiLSTM on Raw Time Series (No Feature Extraction)
Dataset: timenew1_preprocessed_pads_132ch
Classification: Parkinson's vs Healthy
Direct processing of 976 timesteps × 132 channels
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

# Load patient information
print("\nLoading patient information...")
patients_df = pd.read_csv(PATIENTS_INFO)

# Filter for Parkinson's and Healthy only
patients_df = patients_df[patients_df['condition'].isin(["Parkinson's", 'Healthy'])].copy()
print(f"Total patients after filtering: {len(patients_df)}")
parkinsons_count = sum(patients_df['condition'] == "Parkinson's")
healthy_count = sum(patients_df['condition'] == 'Healthy')
print(f"  Parkinson's: {parkinsons_count}")
print(f"  Healthy: {healthy_count}")

# Create binary labels (1 = Parkinson's, 0 = Healthy)
patients_df['label'] = (patients_df['condition'] == "Parkinson's").astype(int)

# Rename PatientID column for consistency
if 'PatientID' in patients_df.columns:
    patients_df = patients_df.rename(columns={'PatientID': 'patient_id'})

# Load channel names
with open(os.path.join(DATA_DIR, 'channel_names.txt'), 'r') as f:
    channel_names = [line.strip() for line in f.readlines()]

print(f"\nDataset Configuration:")
print(f"  Total channels: {N_CHANNELS}")
print(f"  Timesteps per patient: {N_TIMESTEPS}")
print(f"  Total raw values per patient: {N_CHANNELS * N_TIMESTEPS}")

# Load raw time series data
print("\nLoading raw time series data...")
X_raw = []
y_labels = []
ages = []
patient_ids = []

for idx, row in tqdm(patients_df.iterrows(), total=len(patients_df), desc="Loading patients"):
    patient_id = row['patient_id']
    file_path = os.path.join(DATA_DIR, f'{patient_id}_ml.bin')

    if os.path.exists(file_path):
        # Load raw binary data
        data = np.fromfile(file_path, dtype=np.float32)

        # Reshape to (n_channels, n_timesteps)
        data = data.reshape(N_CHANNELS, N_TIMESTEPS)

        X_raw.append(data)
        y_labels.append(row['label'])
        ages.append(row['age_at_diagnosis'])
        patient_ids.append(patient_id)

X_raw = np.array(X_raw)  # Shape: (n_patients, n_channels, n_timesteps)
y_labels = np.array(y_labels)
ages = np.array(ages).reshape(-1, 1)

print(f"\nLoaded data shape: {X_raw.shape}")
print(f"Labels shape: {y_labels.shape}")
print(f"Ages shape: {ages.shape}")

# Normalize raw time series (per channel across all timesteps)
print("\nNormalizing raw time series data...")
X_normalized = np.zeros_like(X_raw)
for i in range(N_CHANNELS):
    channel_data = X_raw[:, i, :].reshape(-1)  # Flatten all patients' data for this channel
    mean = channel_data.mean()
    std = channel_data.std()
    X_normalized[:, i, :] = (X_raw[:, i, :] - mean) / (std + 1e-8)

# Normalize age
age_scaler = StandardScaler()
ages_normalized = age_scaler.fit_transform(ages)

# Train-test split
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
    X_normalized, y_labels, ages_normalized, test_size=0.2, random_state=42, stratify=y_labels
)

print(f"Train set: {X_train.shape[0]} patients")
print(f"Test set: {X_test.shape[0]} patients")

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {class_weights_dict}")

# Convert to torch tensors
class_weights_tensor = torch.FloatTensor(class_weights).to(device)


# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ages):
        # X shape: (n_patients, n_channels, n_timesteps)
        # Transpose to (n_patients, n_timesteps, n_channels) for LSTM
        self.X = torch.FloatTensor(X).transpose(1, 2)
        self.y = torch.LongTensor(y)
        self.ages = torch.FloatTensor(ages)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.ages[idx], self.y[idx]


# BiLSTM Model for Raw Time Series
class RawBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(RawBiLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Age integration
        self.age_fc = nn.Linear(1, 32)

        # Classification layers
        self.fc1 = nn.Linear(hidden_size * 2 + 32, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 2)

        self.relu = nn.ReLU()

    def attention_net(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)

        # Apply attention
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: (batch_size, hidden_size * 2)

        return context_vector

    def forward(self, x, age):
        # x shape: (batch_size, seq_len=976, input_size=132)

        # BiLSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, 976, hidden_size * 2)

        # Apply attention
        context = self.attention_net(lstm_out)
        # context shape: (batch_size, hidden_size * 2)

        # Process age
        age_features = self.relu(self.age_fc(age))
        # age_features shape: (batch_size, 32)

        # Concatenate LSTM context and age features
        combined = torch.cat([context, age_features], dim=1)
        # combined shape: (batch_size, hidden_size * 2 + 32)

        # Classification
        x = self.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(X_train, y_train, age_train)
test_dataset = TimeSeriesDataset(X_test, y_test, age_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model
print("\n" + "="*80)
print("RAW TIME SERIES BiLSTM MODEL")
print("="*80)
print(f"Input: {N_TIMESTEPS} timesteps × {N_CHANNELS} channels (RAW DATA)")
print(f"No feature extraction - direct time series processing")

model = RawBiLSTMModel(input_size=N_CHANNELS, hidden_size=128, num_layers=2, dropout=0.3)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training
print("\nTraining Raw BiLSTM...")
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for X_batch, age_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        age_batch = age_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch, age_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for X_batch, age_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            age_batch = age_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch, age_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    val_acc = accuracy_score(val_labels, val_preds)

    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'bilstm_raw_timeseries_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

# Load best model
print("\nLoading best model...")
model.load_state_dict(torch.load('bilstm_raw_timeseries_best.pth'))

# Final evaluation
print("\nFinal Evaluation on Test Set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, age_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        age_batch = age_batch.to(device)

        outputs = model(X_batch, age_batch)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)

print("\n" + "="*80)
print("RAW TIME SERIES BiLSTM RESULTS")
print("="*80)
print(f"Direct processing of 976 timesteps × 132 channels")
print(f"No feature extraction applied")
print()
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print()
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Healthy', 'Parkinsons']))

# Save results
results_text = f"""================================================================================
RAW TIME SERIES BiLSTM RESULTS
Dataset: PADS Preprocessed (132 channels, 976 timesteps)
Classification: Parkinson's vs Healthy
================================================================================

Model Configuration:
  - Direct processing of raw time series (NO feature extraction)
  - Input: 976 timesteps × 132 channels
  - BiLSTM with attention mechanism
  - Hidden size: 128
  - Number of layers: 2
  - Dropout: 0.3
  - Age at diagnosis included

Dataset Information:
  Total patients: {len(patients_df)}
  Parkinson's: {parkinsons_count}
  Healthy: {healthy_count}
  Class weights: {class_weights_dict}

Results:
  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
  F1 Score:  {f1:.4f} ({f1*100:.2f}%)
  Precision: {precision:.4f} ({precision*100:.2f}%)
  Recall:    {recall:.4f} ({recall*100:.2f}%)

Classification Report:
{classification_report(all_labels, all_preds, target_names=['Healthy', 'Parkinsons'])}

Comparison with Feature-Based BiLSTM:
  Feature-Based BiLSTM (9 windows, 31 features/channel): 84.51%
  Raw Time Series BiLSTM (976 timesteps, no features): {accuracy*100:.2f}%
"""

with open('bilstm_raw_timeseries_results.txt', 'w') as f:
    f.write(results_text)

print("\n" + "="*80)
print("Results saved to: bilstm_raw_timeseries_results.txt")
print("Model saved to: bilstm_raw_timeseries_best.pth")
print("="*80)
