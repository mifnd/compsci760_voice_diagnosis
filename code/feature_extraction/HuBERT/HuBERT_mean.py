#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
root = Path(__file__).resolve().parents[3]
DATA_DIR = root / "data/raw/patient-vocal-dataset"
OUTPUT_FILE = root / "data/processed/hubert_features.csv"

# Load HuBERT model and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model.to(device)
model.eval()

# Constants
TARGET_SR = 16000
classes = ["Normal", "Laryngozele", "Vox_senilis"]

# Collect features
all_data = []
uid = 0

for label in classes:
    folder = DATA_DIR / label
    audio_files = list(folder.glob("*.wav"))

    for file_path in tqdm(audio_files, desc=f"Processing {label}"):
        if "egg" in file_path.name:
            continue
        try:
            # Load and preprocess audio
            waveform, sr = torchaudio.load(file_path)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)
                sr = TARGET_SR

            waveform = waveform.mean(dim=0)  # Convert to mono

            # Extract features
            inputs = feature_extractor(
                waveform.numpy(), sampling_rate=sr, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                output = model(input_values)
                features = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            # Parse metadata
            patient_number = file_path.name.split("-")[0]
            sound_type = file_path.stem.split("-")[-1]

            entry = {
                "id": uid,
                "patient_number": patient_number,
                "disease_label": label,
                "file_name": file_path.name,
                "sound_type": sound_type,
            }
            for i, val in enumerate(features):
                entry[f"feature_{i}"] = val

            all_data.append(entry)
            uid += 1

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

# Save features to CSV
df = pd.DataFrame(all_data)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Feature file saved to: {OUTPUT_FILE}")

