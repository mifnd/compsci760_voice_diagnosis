from xgb_runner import run_xgb_pipeline

if __name__ == "__main__":
    main_csv = "data/processed/10mfcc_mean.csv"
    train_txt = "data/train1.txt"
    test_txt = "data/test1.txt"
    aug_csv_list = []
    run_xgb_pipeline(main_csv, train_txt, test_txt, aug_csv_list)
