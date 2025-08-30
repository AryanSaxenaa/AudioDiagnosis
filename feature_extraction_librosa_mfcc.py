
import librosa
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Configuration ---
AUDIO_DATASET_PATH = 'washing_machine_audio_wav_improved/Washing machine/'
OUTPUT_DATA_FILE = 'extracted_audio_features.pkl'
N_MFCC = 13  # Number of MFCCs to extract
MAX_PAD_LEN = 174 # You might need to adjust this based on your audio files

# --- Function to extract features ---
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)

        # Pad or truncate MFCCs to a fixed length
        if mfccs.shape[1] > MAX_PAD_LEN:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    return mfccs

# --- Main script ---
features = []
labels = []

# Iterate through the main "Abnormal" and "Normal" folders
for label_folder in os.listdir(AUDIO_DATASET_PATH):
    label_path = os.path.join(AUDIO_DATASET_PATH, label_folder)
    if os.path.isdir(label_path):
        # The label is "Abnormal" or "Normal"
        label = label_folder.split(' - ')[1] 
        
        # Iterate through the subfolders (e.g., "Background noise", "Dehydration mode noise")
        for subfolder in os.listdir(label_path):
            subfolder_path = os.path.join(label_path, subfolder)
            if os.path.isdir(subfolder_path):
                
                # Iterate through the .wav files
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(subfolder_path, filename)
                        
                        # Extract features
                        data = extract_features(file_path)
                        
                        if data is not None:
                            features.append(data)
                            labels.append(label)

# Convert to numpy arrays
features = np.array(features)

# Encode the labels (e.g., "Abnormal" -> 0, "Normal" -> 1)
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Save the extracted data and the label encoder
with open(OUTPUT_DATA_FILE, 'wb') as f:
    pickle.dump({
        'features': features,
        'labels': encoded_labels,
        'label_encoder': le
    }, f)

print(f"Feature extraction complete. Data saved to {OUTPUT_DATA_FILE}")
print(f"Shape of features array: {features.shape}")
print(f"Labels: {le.classes_}")