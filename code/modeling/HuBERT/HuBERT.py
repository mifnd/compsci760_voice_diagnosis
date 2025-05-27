from transformers import HubertModel, HubertConfig, Wav2Vec2FeatureExtractor
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Base directory
root = Path(__file__).resolve().parents[3]
DATA_DIR = root / "data/raw/patient-vocal-dataset"
OUTPUT_FILE = root / "data/processed/hubert_self_attentive_features.csv"

# Load HuBERT and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model.eval()

# Self-attentive pooling
def self_attentive_pooling(hidden_states):
    weights = torch.nn.functional.softmax(hidden_states.mean(dim=-1), dim=1).unsqueeze(-1)
    return (hidden_states * weights).sum(dim=1)

# Collect data
data = []
sound_types = ["Normal", "Laryngozele", "Vox_senilis"]

for label in sound_types:
    label_dir = DATA_DIR / label
    audio_files = list(label_dir.glob("*.wav"))

    for i, file_path in enumerate(tqdm(audio_files, desc=f"Processing {label}")):
        try:
            waveform, sample_rate = torchaudio.load(file_path)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state

            pooled = self_attentive_pooling(hidden_states)
            pooled_np = pooled.squeeze().numpy()

            row = {
                "id": len(data),
                "patient_number": f"patient_{i+1}",
                "disease_label": label,
                "file_name": file_path.name,
                "sound_type": label,
            }

            for j, val in enumerate(pooled_np):
                row[f"feature_{j}"] = val

            data.append(row)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

# Save to CSV
df = pd.DataFrame(data)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Feature file saved to: {OUTPUT_FILE}")
