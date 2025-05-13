from xgb_runner import run_xgb_pipeline

if __name__ == "__main__":
    # real data paths
    real_paths = ["data/processed/10mfcc_mean.csv"]
    # augmented data paths
    aug_paths = [
        "data/processed/aug_pitch_down_10mfcc_mean.csv",
        "data/processed/aug_pitch_up_10mfcc_mean.csv",
    ]
    run_xgb_pipeline(real_paths, aug_paths)
