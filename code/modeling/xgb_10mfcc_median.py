from xgb_runner import run_xgb_pipeline

if __name__ == "__main__":
    # real data paths
    real_paths = ["data/processed/10mfcc_median.csv"]
    # augmented data paths
    aug_paths = [
        ]
    run_xgb_pipeline(real_paths, aug_paths)
