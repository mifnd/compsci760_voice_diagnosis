from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

root = Path("../..").resolve()
data_path = root / "data/processed/10mfcc_mean.csv"

voice_df = pd.read_csv(data_path, index_col=0)
mfcc_df = voice_df.drop(columns=["patient_number", "disease_label", "file_name", "sound_type", "is_egg"])

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
mfcc_tsne = tsne.fit_transform(mfcc_df)

mfcc_tsne_wide_df = pd.DataFrame({
    "mfcc_tsne1": mfcc_tsne[:, 0],
    "mfcc_tsne2": mfcc_tsne[:, 1],
    "disease_label": voice_df["disease_label"],
    "patient_number": voice_df["patient_number"],
    "sound_type": voice_df["sound_type"]
})

# Make plots and save in plots folder
plot = sns.scatterplot(data=mfcc_tsne_wide_df, x="mfcc_tsne1", y="mfcc_tsne2", hue="disease_label")\
    .set_title(label="t-SNE of the 10 first MFCCs")
plt.savefig(root / "plots/data_exploration/mfcc_tsne_disease.png", dpi=200)
plt.clf()

plot2 = sns.scatterplot(data=mfcc_tsne_wide_df, x="mfcc_tsne1", y="mfcc_tsne2", hue="patient_number", palette="deep")\
    .set_title(label="t-SNE of the 10 first MFCCs")
plt.savefig(root / "plots/data_exploration/mfcc_tsne_patient.png", dpi=200)
plt.clf()

plot = sns.scatterplot(data=mfcc_tsne_wide_df, x="mfcc_tsne1", y="mfcc_tsne2", hue="sound_type")\
    .set_title(label="t-SNE of the 10 first MFCCs")
plt.savefig(root / "plots/data_exploration/mfcc_tsne_sound.png", dpi=200)
plt.clf()
