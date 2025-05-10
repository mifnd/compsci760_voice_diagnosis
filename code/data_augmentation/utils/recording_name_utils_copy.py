from pathlib import Path

# This file contains method for extracting patient information from the recording file name (path).


def get_is_egg(file_path: Path):
    file_stem_parts = file_path.stem.split("-")
    return len(file_stem_parts) > 2 and file_stem_parts[2] == "egg"