from pathlib import Path
from fe_mfcc import create_csv_with_mfcc

# Compute mean of the first 10 mfcc vectors for the dataset "aug_pitch_down".
root = Path("../..").resolve()
recordings_path = root / "data/interim/aug_pitch_down"
csv_output_path = root / "data/processed/aug_pitch_down_10mfcc_mean.csv"

create_csv_with_mfcc(recordings_path=recordings_path,
                     csv_output_path=csv_output_path,
                     n_mfcc=10,
                     avg_type="mean")
