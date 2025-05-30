import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Base path
root = Path("../..").resolve()
FEATURE_FILE = root / "data/processed/hubert_self_attentive_features.csv"
OUTPUT_IMAGE = root / "plots/data_exploration/HubertSA_tsne_visualization.png"

# Load features
df = pd.read_csv(FEATURE_FILE)
feature_cols = [col for col in df.columns if col.startswith("feature_")]
X = df[feature_cols].values
y = df["disease_label"]

# Apply t-SNE
print("Performing t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_tsne = tsne.fit_transform(X)

# Build visualization DataFrame
vis_df = pd.DataFrame({
    "x": X_tsne[:, 0],
    "y": X_tsne[:, 1],
    "label": y
})

# Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=vis_df, x="x", y="y", hue="label", palette="Set2", s=60, alpha=0.85)
plt.title("t-SNE visualization of HuBERT + SA features", fontsize=16)
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend(title="Disease Type", fontsize=10)
plt.grid(True)
plt.tight_layout()

OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.show()

