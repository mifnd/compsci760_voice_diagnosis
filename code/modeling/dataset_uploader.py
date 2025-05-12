from xgb_runner import run_xgb_pipeline

paths = [
    "data/processed/10mfcc_mean.csv",
    "data/processed/aug_pitch_down_10mfcc_mean.csv",
    "data/processed/aug_pitch_up_10mfcc_mean.csv",
]
run_xgb_pipeline(paths)
