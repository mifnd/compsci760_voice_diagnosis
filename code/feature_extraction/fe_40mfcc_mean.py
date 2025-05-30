from pathlib import Path
from fe_mfcc import create_csv_with_mfcc

root = Path("../..").resolve()
recordings_path = root / "data/raw/patient-vocal-dataset"
csv_output_path = root / "data/processed/40mfcc_mean.csv"

create_csv_with_mfcc(recordings_path=recordings_path,
                     csv_output_path=csv_output_path,
                     n_mfcc=40,
                     avg_type="mean")
