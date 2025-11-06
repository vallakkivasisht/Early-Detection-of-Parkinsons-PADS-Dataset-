"""
Preprocess timenew1 with PADS-style preprocessing (132 channels)
Output: 132 channels (11 tasks × 2 wrists × 2 sensors × 3 axes)
Method: L1 trend filtering on accelerometer, trim first 48 samples

This uses ALL 11 available tasks (including LiftHold, PointFinger, TouchIndex)
to create a 132-channel dataset closer to the original PADS preprocessing.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# L1 trend filter implementation
def l1_trend_filter(y, vlambda=50.0, verbose=False):
    """
    Apply L1 trend filtering to remove drift from signal.

    Args:
        y: Input signal (1D array)
        vlambda: Regularization parameter (higher = smoother trend)
        verbose: Print convergence info

    Returns:
        Trend signal
    """
    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except ImportError:
        print("Warning: scipy not available, skipping L1 trend filtering")
        return np.zeros_like(y)

    n = len(y)

    # Build second-order difference matrix
    e = np.ones(n)
    D = sparse.diags([e, -2*e, e], [0, 1, 2], shape=(n-2, n))

    # Solve using iterative reweighting
    w = np.ones(n)
    for iteration in range(5):  # 5 iterations usually sufficient
        W = sparse.diags(w, 0)
        C = W + vlambda * D.T @ D
        trend = spsolve(C, w * y)

        # Update weights
        residual = y - trend
        w = 1.0 / (np.abs(residual) + 1e-6)

    return trend


print("=" * 80)
print("Preprocessing timenew1 to 132 Channels (PADS-style - All 11 tasks)")
print("=" * 80)

# Define ALL 11 tasks (including LiftHold, PointFinger, TouchIndex)
TASKS = [
    'Relaxed',
    'RelaxedTask',
    'StretchHold',
    'LiftHold',       # Included (PADS-style)
    'HoldWeight',
    'PointFinger',    # Included (PADS-style)
    'DrinkGlas',
    'CrossArms',
    'TouchIndex',     # Included (PADS-style)
    'TouchNose',
    'Entrainment'
]

WRISTS = ['LeftWrist', 'RightWrist']
SENSORS = ['Accelerometer', 'Gyroscope']
AXES = ['X', 'Y', 'Z']

# Calculate expected channels
N_CHANNELS = len(TASKS) * len(WRISTS) * len(SENSORS) * len(AXES)
print(f"\nExpected channels: {N_CHANNELS}")
print(f"  {len(TASKS)} tasks × {len(WRISTS)} wrists × {len(SENSORS)} sensors × {len(AXES)} axes")

# Load patient info
print("\n[1/4] Loading patient information...")
patient_info = pd.read_csv('patients_info_encoded.csv')
print(f"Total patients: {len(patient_info)}")

# Create output directory
output_dir = 'timenew1_preprocessed_pads_132ch'
Path(output_dir).mkdir(exist_ok=True)

# Process each patient
print("\n[2/4] Processing patients...")
data_dir = 'timenew1'

successful = 0
failed = []

for idx, row in tqdm(patient_info.iterrows(), total=len(patient_info)):
    patient_id = str(row['PatientID']).zfill(3)
    patient_path = os.path.join(data_dir, patient_id)

    if not os.path.exists(patient_path):
        failed.append(f"{patient_id} (directory not found)")
        continue

    try:
        # Collect data for all task/wrist/sensor combinations
        all_channels = []
        channel_names = []

        # Build channel order: Task -> Wrist -> Sensor -> Axis
        for task in TASKS:
            for wrist in WRISTS:
                for sensor in SENSORS:
                    for axis in AXES:
                        channel_names.append(f"{task}_{wrist}_{sensor}_{axis}")

        # Load CSV files and organize data
        min_length = None
        channel_data = {}

        for task in TASKS:
            for wrist in WRISTS:
                csv_file = f"{patient_id}_{task}_{wrist}.csv"
                csv_path = os.path.join(patient_path, csv_file)

                if not os.path.exists(csv_path):
                    # File doesn't exist, skip this patient
                    raise FileNotFoundError(f"Missing {csv_file}")

                # Load CSV (format: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
                df = pd.read_csv(csv_path, header=None)

                # Extract sensor data (columns 1-7, skip timestamp at column 0)
                if df.shape[1] < 7:
                    raise ValueError(f"Invalid CSV format in {csv_file}")

                # Update minimum length
                if min_length is None:
                    min_length = len(df)
                else:
                    min_length = min(min_length, len(df))

                # Store data for each axis
                for sensor_idx, sensor in enumerate(SENSORS):
                    if sensor == 'Accelerometer':
                        cols = [1, 2, 3]  # acc_x, acc_y, acc_z
                    else:  # Gyroscope
                        cols = [4, 5, 6]  # gyro_x, gyro_y, gyro_z

                    for axis_idx, axis in enumerate(AXES):
                        channel_name = f"{task}_{wrist}_{sensor}_{axis}"
                        channel_data[channel_name] = df.iloc[:, cols[axis_idx]].values

        # Trim all channels to same length
        for channel_name in channel_names:
            if channel_name in channel_data:
                channel_data[channel_name] = channel_data[channel_name][:min_length]

        # Stack into array (channels × timesteps)
        data_array = np.array([channel_data[name] for name in channel_names])

        # Verify channel count
        if data_array.shape[0] != N_CHANNELS:
            raise ValueError(f"Expected {N_CHANNELS} channels, got {data_array.shape[0]}")

        # [3/4] Apply L1 trend filtering to accelerometer channels ONLY
        print(f"  Patient {patient_id}: Applying L1 trend filtering...", end='\r')
        acc_mask = np.array(['Accelerometer' in name for name in channel_names])

        for i in range(data_array.shape[0]):
            if acc_mask[i]:
                trend = l1_trend_filter(data_array[i, :], vlambda=50.0)
                data_array[i, :] = data_array[i, :] - trend

        # [4/4] Trim first 48 samples (0.48 seconds vibration artifact)
        if data_array.shape[1] > 48:
            data_array = data_array[:, 48:]
        else:
            # Signal too short
            raise ValueError(f"Signal too short: {data_array.shape[1]} samples")

        # Save as binary file
        output_path = os.path.join(output_dir, f"{patient_id}_ml.bin")
        data_array.astype(np.float32).tofile(output_path)

        successful += 1

    except Exception as e:
        failed.append(f"{patient_id} ({str(e)})")
        continue

print(f"\n[3/4] Preprocessing complete!")
print(f"Successful: {successful}/{len(patient_info)}")
print(f"Failed: {len(failed)}")

if len(failed) > 0 and len(failed) <= 10:
    print("\nFailed patients:")
    for f in failed:
        print(f"  - {f}")
elif len(failed) > 10:
    print(f"\nFirst 10 failed patients:")
    for f in failed[:10]:
        print(f"  - {f}")

# Verify output
if successful > 0:
    print(f"\n[4/4] Verifying output...")
    sample_file = os.listdir(output_dir)[0]
    sample_path = os.path.join(output_dir, sample_file)
    sample_data = np.fromfile(sample_path, dtype=np.float32)

    n_channels = N_CHANNELS
    n_timesteps = len(sample_data) // n_channels

    print(f"Sample file: {sample_file}")
    print(f"  Total values: {len(sample_data)}")
    print(f"  Channels: {n_channels}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Shape: ({n_channels}, {n_timesteps})")
    print(f"\nOutput directory: {output_dir}/")
    print(f"Format: {patient_id}_ml.bin (float32 binary)")

    # Save channel names for reference
    channel_names_file = os.path.join(output_dir, 'channel_names.txt')
    with open(channel_names_file, 'w') as f:
        for name in channel_names:
            f.write(f"{name}\n")
    print(f"Channel names saved to: {channel_names_file}")

print("\n" + "=" * 80)
print("PADS-STYLE PREPROCESSING COMPLETED!")
print(f"Dataset: 132 channels (11 tasks)")
print(f"Preprocessing:")
print(f"  - L1 trend filtering (λ=50) on accelerometer only")
print(f"  - Vibration artifact removal (first 48 samples)")
print(f"  - All 11 tasks included (no task exclusion)")
print("=" * 80)
