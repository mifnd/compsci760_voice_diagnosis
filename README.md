# Diagnosis of Throat Disorders with Voice Recordings using ML
Welcome to the GitHub of the throat disorders project from 
CompSci760 at University of Auckland. The aim of the project is to build 
machine learning models that can predict throat disorders
purely from voice recordings of the patients. Follow along on the exciting journey.

## Setup
We use Conda (Anaconda or Miniconda) to manage our python environment. In order to
set up an environment with all the correct packages, cd into the folder with the 
`environments.yml` file and run

`conda env create --name <YourEnvironmentName> --file environment.yml`

Activate your environment. You should now be able to run all scripts in the project. 

If the `environment.yml` gets updated with new packages, you can use the 
following command to update your conda-environment:

`conda env update --file environment.yml --prune`

## Structure of the project
The project is divided into multiple folders.
- The `code` folder contains all scripts for data exploration, data augmentation, 
feature extraction and modeling. 
- The `data` folder contains all our data. In the subfolder `raw`, you will find the original 
dataset consisting of voice recordings. The subfolder `interim` contains augmented datasets - 
i.e. the original recordings with added noise, with the pitch shifted up or down etc. 
Finally, the `processed` subfolder contains .csv-files of numeric feature values extracted from
the original dataset and the augmented versions.
- The `plots` folder contains various plots for data exploration.
- The `results` folder contains logs of results from all the different experiments. 

## Experiments and naming conventions
All scripts and datasets are named such that it (hopefully) is easy to track them through the pipelines.
Here are two examples to help understand both the naming and the process of our experiments:
- Consider the original voice recordings in the folder `data/raw`. The script 
`code/feature_extraction/fe_10mfcc_mean.py` extracts the mean of the 10 first MFCCs for all recordings.
The result is saved as the .csv-file `data/processed/10mfcc_mean.csv`. The dataset is used in multiple
experiments - e.g. in the training of a simple XGBoost model in the script 
`code/modeling/xbg_10mfcc_mean.py`. 


- Let's now consider an example using an augmented dataset. The script 
`code/data_augmentation/data_aug_pitch_up.py` will fetch the original voice recordings, change the pitch 
and save the new recordings in the folder `data/interim/aug_pitch_up`. The script 
`code/feature_extraction/fe_aug_pitch_up_10mfcc_mean.py` computes the mean of the 10 first MFCCs and saves
the result as a .csv-file in `data/processed/aug_pitch_up_10mfcc_mean.csv`. This dataset is for example 
used in the training of an XGBoost model in `code/modeling/xgb_10mfcc_mean_aug_pitch_up.py`.
