from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

root = Path("../..").resolve()
data_path = root / "data/processed/10mfcc_mean.csv"

voice_df = pd.read_csv(data_path, index_col=0)
mfcc_df = voice_df.drop(columns=["patient_number", "disease_label", "file_name", "sound_type", "is_egg"])

pca = PCA(n_components=2)
pca.fit(mfcc_df)
mfcc_pca = pca.transform(mfcc_df)

mfcc_pca_wide_df = pd.DataFrame({
    "mfcc_pca1": mfcc_pca[:, 0],
    "mfcc_pca2": mfcc_pca[:, 1],
    "disease_label": voice_df["disease_label"],
    "patient_number": voice_df["patient_number"],
    "sound_type": voice_df["sound_type"]
})

# Make plots and save in plots folder
plot = sns.scatterplot(data=mfcc_pca_wide_df, x="mfcc_pca1", y="mfcc_pca2", hue="disease_label")\
    .set_title(label="PCA of the 10 first MFCCs")
plt.savefig(root / "plots/data_exploration/mfcc_pca_disease.png", dpi=200)
plt.clf()

plot2 = sns.scatterplot(data=mfcc_pca_wide_df, x="mfcc_pca1", y="mfcc_pca2", hue="patient_number", palette="deep")\
    .set_title(label="PCA of the 10 first MFCCs")
plt.savefig(root / "plots/data_exploration/mfcc_pca_patient.png", dpi=200)
plt.clf()

plot = sns.scatterplot(data=mfcc_pca_wide_df, x="mfcc_pca1", y="mfcc_pca2", hue="sound_type")\
    .set_title(label="PCA of the 10 first MFCCs")
plt.savefig(root / "plots/data_exploration/mfcc_pca_sound.png", dpi=200)
plt.clf()
