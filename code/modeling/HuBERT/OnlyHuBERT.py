#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import torch
import torchaudio
import pandas as pd
from transformers import HubertModel, Wav2Vec2FeatureExtractor


# In[19]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the model and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960", cache_dir="D:/huggingface_cache")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", cache_dir="D:/huggingface_cache")
model.eval()


# In[28]:


TARGET_SR = 16000  #HuBERT Required Sampling Rate

base_dir = "patient-vocal-dataset"
classes = ["Normal", "Laryngozele", "Vox_senilis"]

all_data = []
uid = 0

for label in classes:
    folder = os.path.join(base_dir, label)
    for fname in os.listdir(folder):
        if fname.endswith(".wav") and "egg" not in fname:
            full_path = os.path.join(folder, fname)

            # Load audio
            waveform, sr = torchaudio.load(full_path)

            # Resample (if not 16000Hz)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)
                sr = TARGET_SR

            # Convert to mono
            waveform = waveform.mean(dim=0)

            # Use feature extractor for padding + normalization
            inputs = feature_extractor(
                waveform.numpy(), sampling_rate=sr, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)

            # Get the output of HuBERT
            with torch.no_grad():
                output = model(input_values)
                features = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            # Parse file name information
            patient_number = fname.split("-")[0]
            sound_type = fname.replace(".wav", "").split("-")[-1]

            entry = {
                "id": uid,
                "patient_number": patient_number,
                "disease_label": label,
                "file_name": fname,
                "sound_type": sound_type,
            }
            for i in range(len(features)):
                entry[f"feature_{i}"] = features[i]

            all_data.append(entry)
            uid += 1


# In[29]:


# Save as CSV
df = pd.DataFrame(all_data)
df.to_csv("hubert_features.csv", index=False)


# In[30]:


import pandas as pd
df = pd.read_csv("hubert_features.csv")


# In[34]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Extract features and labels
feature_cols = [col for col in df.columns if col.startswith("feature_")]
X = df[feature_cols].values
y = df["disease_label"]

# ➤ Method 1: PCA (faster)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualization
vis_df = pd.DataFrame({
    "x": X_pca[:, 0],
    "y": X_pca[:, 1],
    "label": y
})

plt.figure(figsize=(10, 7))
sns.scatterplot(data=vis_df, x="x", y="y", hue="label", palette="Set2", s=60, alpha=0.8)
plt.title("PCA visualization of HuBERT feature space")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Disease Label")
plt.grid(True)
plt.show()


# In[36]:


# ➤ Method 2: t-SNE (more non-linear, slower but more natural)
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved feature file
df = pd.read_csv("hubert_features.csv")

# Extract feature vectors and labels
feature_cols = [col for col in df.columns if col.startswith("feature_")]
X = df[feature_cols].values
y = df["disease_label"]

# t-SNE Dimensionality Reduction (2D)
print("Performing t-SNE dimensionality reduction (this may take some time)...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_tsne = tsne.fit_transform(X)

# Building a Visualization DataFrame
vis_df = pd.DataFrame({
    "x": X_tsne[:, 0],
    "y": X_tsne[:, 1],
    "label": y
})

# Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=vis_df, x="x", y="y", hue="label", palette="Set2", s=60, alpha=0.85)
plt.title("t-SNE visualization of HuBERT features", fontsize=16)
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend(title="Disease Type", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("hubert_tsne_visualization.png", dpi=300)
plt.show()


# In[ ]:




