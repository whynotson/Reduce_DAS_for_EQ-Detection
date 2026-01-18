"""Run inference on DAS-Reduce dataset"""

import numpy as np
from tensorflow.keras.models import load_model
import os

# Paths
MODEL_PATH = '/storage/student8/LightEQ/lighteq_original_test_outputs/final_model.h5'
DATA_DIR = '/storage/student8/LightEQ_DAS_Reduce/preprocessed'


print("="*80)
print("INFERENCE ON DAS-REDUCE DATASET")
print("="*80)

# Load model
print("\n📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# Load data
print("\n📂 Loading preprocessed data...")
X_das = np.load(os.path.join(DATA_DIR, 'X_das_reduced.npy'))
metadata = np.load(os.path.join(DATA_DIR, 'metadata_reduced.npy'), allow_pickle=True)
print(f"✅ Data loaded: {X_das.shape}")
print(f"   Files: {len(metadata)}")

# Run inference
print("\n🔮 Running inference...")
predictions = model.predict(X_das, batch_size=16, verbose=1)
print(f"✅ Predictions shape: {predictions.shape}")

# Handle multi-timestep predictions
if len(predictions.shape) == 3:
    # Shape: (batch, timesteps, 1) -> Take max across timesteps
    predictions_max = predictions.max(axis=1)  # (batch, 1)
    predictions = predictions_max
    print(f"   Reduced to: {predictions.shape} (max across timesteps)")

# Flatten if needed
if len(predictions.shape) == 2:
    predictions = predictions.flatten()  # (batch,)

# Apply 1000x calibration
predictions_scaled = predictions * 1000

print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)
print(f"Raw predictions:")
print(f"  Min:  {predictions.min():.6f}")
print(f"  Max:  {predictions.max():.6f}")
print(f"  Mean: {predictions.mean():.6f}")
print(f"  Std:  {predictions.std():.6f}")

print(f"\nScaled predictions (×1000):")
print(f"  Min:  {predictions_scaled.min():.4f}")
print(f"  Max:  {predictions_scaled.max():.4f}")
print(f"  Mean: {predictions_scaled.mean():.4f}")
print(f"  Std:  {predictions_scaled.std():.4f}")

# Per-file predictions
print("\n" + "="*80)
print("PREDICTIONS BY FILE")
print("="*80)
for i, meta in enumerate(metadata):
    file_short = meta['file'].split('_')[2] + '_' + meta['file'].split('_')[3][:8]
    raw_val = predictions[i]
    scaled_val = predictions_scaled[i]
    status = "🟢 EVENT" if scaled_val > 0.05 else "🔴 NOISE"
    print(f"{file_short}: raw={raw_val:.6f}, scaled={scaled_val:.4f} {status}")

# Count detections
threshold = 0.05
detected = np.sum(predictions_scaled > threshold)
print(f"\n{'='*80}")
print(f"DETECTION SUMMARY (threshold={threshold})")
print(f"{'='*80}")
print(f"Detected: {detected}/13 events ({detected*100/13:.1f}%)")
print(f"Missed: {13-detected}/13 events ({(13-detected)*100/13:.1f}%)")

# Save
np.save(os.path.join(DATA_DIR, 'predictions_reduced.npy'), predictions)
np.save(os.path.join(DATA_DIR, 'predictions_scaled_reduced.npy'), predictions_scaled)

print(f"\n💾 Saved:")
print(f"   {DATA_DIR}/predictions_reduced.npy")
print(f"   {DATA_DIR}/predictions_scaled_reduced.npy")