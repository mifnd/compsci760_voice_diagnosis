from pathlib import Path
from fe_mfcc_mean import create_csv_with_mfcc_mean

# Compute mean of the first 10 mfcc vectors for the dataset "aug_noise".
root = Path("../..").resolve()
recordings_path = root / "data/interim/aug_noise"
csv_output_path = root / "data/processed/aug_noise_10mfcc_mean.csv"

create_csv_with_mfcc_mean(recordings_path=recordings_path,
                          csv_output_path=csv_output_path,
                          n_mfcc=10)
