# import packages
import pandas as pd
from pathlib import Path

# Set default path to folder with the Laryngozele, Normal, and Vox Senilis folders
data_path = Path("./data/raw/patient-vocal-dataset")

# Find the paths to each of the Laryngozele, Normal, and Vox Senilis folders and store their names to use as labels
disease_paths = [x for x in data_path.iterdir() if x.is_dir()]
labels = [x.parts[-1] for x in disease_paths]


# innitialise lists
recording_paths = []
recording_names = []
recoding_labels = []
patient_number = []
sound_type = []
is_egg = []

# visit each file and record their path, name, and label
for i in range(len(disease_paths)):
    for file in disease_paths[i].iterdir():
        if not file.parts[-1].find(".wav") == -1: # Ignore all non .wav files
            recording_paths.append(str(file))
            recording_names.append(file.parts[-1])
            recoding_labels.append(labels[i])


# extract patient number, sound type, and is_egg information from each file name
for i in range(len(recording_names)):
    string = recording_names[i].split("-")
    string[len(string) - 1] = string[len(string) - 1].replace(".wav", "")
    if len(string) < 3: string.append("normal")
    patient_number.append(string[0])
    sound_type.append(string[1])
    is_egg.append(string[2] == "egg")

# compile everything in a pandas data frame
df_voice_data = pd.DataFrame({"id":pd.Series(range(len(recoding_labels))), 
              "label":pd.Series(recoding_labels), 
              "file_name":pd.Series(recording_names), 
              "file_path":pd.Series(recording_paths),
              "patient_number":pd.Series(patient_number),
              "sound_type":pd.Series(sound_type),
              "is_egg":pd.Series(is_egg)})

# write to .csv file
df_voice_data.to_csv("recording_data.csv", index=False)