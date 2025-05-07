from pathlib import Path

# This file contains method for extracting patient information from the recording file name (path).


def get_patient_number(file_path: Path):
    file_stem_parts = file_path.stem.split("-")
    return file_stem_parts[0]


def get_sound_type(file_path: Path):
    file_stem_parts = file_path.stem.split("-")
    return file_stem_parts[1]


def get_is_egg(file_path: Path):
    file_stem_parts = file_path.stem.split("-")
    return len(file_stem_parts) > 2 and file_stem_parts[2] == "egg"
