"""
Preprocess DAS-BIGORRE reduced dataset (EQ + Noise only)
Structure: data=(1896, 4400), distance, time
"""

import h5py
import numpy as np
from scipy import signal
from scipy.signal import stft
import os
from tqdm import tqdm

# Paths
DATA_DIR = '/storage/student8/LightEQ_DAS_Reduce/preprocessed'
OUTPUT_DIR = '/storage/student8/LightEQ_DAS_Reduce/preprocessed'

# Parameters
TARGET_FS = 100  # Hz
WINDOW_SIZE = 60  # seconds
OVERLAP = 0.5
NPERSEG = 80

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_das_files():
    """Get all processed DAS files"""
    files = [f for f in os.listdir(DATA_DIR) 
             if f.endswith('_processed.h5')]
    files.sort()
    return files

def preprocess_das_file(filepath):
    """Preprocess single DAS file"""
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]  # Shape: (1896, 4400)
        
        # Determine if (channels, time) or (time, channels)
        shape = data.shape
        print(f"  Original shape: {shape}")
        
        # If first dim >> second dim, likely (time, channels)
        if shape[0] > shape[1]:
            # (1896, 4400) -> 1896 < 4400, so this is (channels, time)
            n_channels, n_samples = shape
        else:
            # Transpose if needed
            n_samples, n_channels = shape
            data = data.T
        
        # With 4400 samples, estimate fs
        # If noise reduced to ~17s event window, fs likely 250Hz
        # 4400 / 250 = 17.6s (matches picks_summary event windows)
        original_fs = 250
        duration = n_samples / original_fs
        
        print(f"  Channels: {n_channels}, Samples: {n_samples}")
        print(f"  Estimated duration: {duration:.2f}s at {original_fs}Hz")
        
        # Select 3 representative channels
        if n_channels >= 3:
            step = max(1, n_channels // 3)
            selected_channels = [step, n_channels//2, n_channels - step]
        else:
            selected_channels = list(range(min(3, n_channels)))
        
        data_selected = data[selected_channels, :]
        print(f"  Selected channels: {selected_channels}")
        
        # Resample to 100Hz
        if original_fs != TARGET_FS:
            num_samples_new = int(n_samples * TARGET_FS / original_fs)
            data_resampled = signal.resample(data_selected, num_samples_new, axis=1)
            print(f"  Resampled: {original_fs}Hz → {TARGET_FS}Hz")
            print(f"  New shape: {data_resampled.shape}")
        else:
            data_resampled = data_selected
        
        return data_resampled, TARGET_FS

def create_windows(data, fs, window_size=60, overlap=0.5):
    """Create sliding windows - but for short files, use whole file"""
    n_channels, n_samples = data.shape
    duration = n_samples / fs
    
    print(f"  Duration: {duration:.2f}s")
    
    # If file shorter than window, pad or use as single window
    if duration < window_size:
        # Use whole file as single window, pad if needed
        window_samples = int(window_size * fs)
        
        if n_samples < window_samples:
            # Pad to window_size
            pad_samples = window_samples - n_samples
            data_padded = np.pad(data, ((0, 0), (0, pad_samples)), mode='constant')
            print(f"  Padded: {n_samples} → {window_samples} samples")
            
            windows = [data_padded]
            metadata = [{
                'window_id': 0,
                'start_sample': 0,
                'end_sample': n_samples,
                'start_time': 0,
                'end_time': duration,
                'padded': True
            }]
        else:
            # Use first window_size seconds
            windows = [data[:, :window_samples]]
            metadata = [{
                'window_id': 0,
                'start_sample': 0,
                'end_sample': window_samples,
                'start_time': 0,
                'end_time': window_size,
                'padded': False
            }]
        
        return windows, metadata
    
    # Normal sliding window for longer files
    window_samples = int(window_size * fs)
    step_samples = int(window_samples * (1 - overlap))
    
    windows = []
    metadata = []
    
    n_windows = (n_samples - window_samples) // step_samples + 1
    
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        
        if end > n_samples:
            break
        
        window_data = data[:, start:end]
        windows.append(window_data)
        
        metadata.append({
            'window_id': i,
            'start_sample': start,
            'end_sample': end,
            'start_time': start / fs,
            'end_time': end / fs,
            'padded': False
        })
    
    return windows, metadata

def compute_stft(window_data, fs):
    """Compute STFT for 3 channels"""
    stft_result = []
    
    for ch in range(window_data.shape[0]):
        f, t, Zxx = stft(window_data[ch, :], 
                         fs=fs, 
                         nperseg=NPERSEG,
                         noverlap=NPERSEG//2)
        stft_result.append(np.abs(Zxx))
    
    # Stack: (freq, time, channels)
    stft_combined = np.stack(stft_result, axis=-1)
    return stft_combined

def normalize_stft(stft_data):
    """Max normalization"""
    max_val = np.max(np.abs(stft_data))
    if max_val > 0:
        return stft_data / max_val
    return stft_data

def main():
    print("="*80)
    print("PREPROCESSING DAS-BIGORRE REDUCED DATASET")
    print("="*80)
    
    files = get_das_files()
    print(f"\nFound {len(files)} processed files")
    
    all_windows = []
    all_metadata = []
    
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"\n📁 {filename}")
        
        # Preprocess
        data, fs = preprocess_das_file(filepath)
        
        # Create windows
        windows, metadata = create_windows(data, fs, WINDOW_SIZE, OVERLAP)
        print(f"  ✅ Created {len(windows)} window(s)")
        
        # Compute STFT and normalize
        for i, (window, meta) in enumerate(zip(windows, metadata)):
            stft_data = compute_stft(window, fs)
            stft_norm = normalize_stft(stft_data)
            
            all_windows.append(stft_norm)
            all_metadata.append({
                'file': filename,
                'window_id': meta['window_id'],
                'start_time': meta['start_time'],
                'end_time': meta['end_time'],
                'padded': meta.get('padded', False)
            })
    
    # Convert to numpy
    X_das = np.array(all_windows)
    
    print(f"\n{'='*80}")
    print(f"✅ PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total windows: {len(X_das)}")
    print(f"Shape: {X_das.shape} (expected: (N, 41, 151, 3))")
    print(f"Stats:")
    print(f"  Mean: {X_das.mean():.6f}")
    print(f"  Std:  {X_das.std():.6f}")
    print(f"  Min:  {X_das.min():.6f}")
    print(f"  Max:  {X_das.max():.6f}")
    
    # Save
    np.save(os.path.join(OUTPUT_DIR, 'X_das_reduced.npy'), X_das)
    np.save(os.path.join(OUTPUT_DIR, 'metadata_reduced.npy'), all_metadata)
    
    print(f"\n💾 Saved:")
    print(f"   {OUTPUT_DIR}/X_das_reduced.npy")
    print(f"   {OUTPUT_DIR}/metadata_reduced.npy")
    
    # Distribution
    print(f"\n{'='*80}")
    print("WINDOW DISTRIBUTION:")
    print(f"{'='*80}")
    file_counts = {}
    for meta in all_metadata:
        file_counts[meta['file']] = file_counts.get(meta['file'], 0) + 1
    
    for file, count in sorted(file_counts.items()):
        short_name = file.split('_')[2] + '_' + file.split('_')[3][:8]
        print(f"  {short_name}: {count} window(s)")

if __name__ == '__main__':
    main()
