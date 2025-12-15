# Utils for converting from Windows training data to Linux training data

import os
import sys
import subprocess
from typing import List, Tuple
import re



WIN_DRIVES_FOLDER: str = "/windrives"

WIN_PATHS_SUFFIX: str = ".txt"
LIN_PATHS_SUFFIX: str = ".lin.txt"


def replace_ignore_case(text: str, old: str, new: str) -> str:
    return re.sub(old, new, text, flags=re.IGNORECASE)


def file_exists(path: str, ignore_case: bool = False) -> bool:
    if not ignore_case:
        return os.path.isfile(path)

    # Case-insensitive mode
    directory, filename = os.path.split(path)
    filename_lower = filename.lower()

    try:
        return filename_lower in (f.lower() for f in os.listdir(directory))
    except FileNotFoundError:
        return False

def try_fix_path(path: str) -> str:
    # fix common path issues, such as case sensitivity

    if(os.path.isfile(path)):
        return path


    # path = replace_ignore_case(path, "images", "images")   # normalize
    # path = replace_ignore_case(path, "train", "train")
    path = replace_ignore_case(path, "/videos/", "/Videos/")

    if(not os.path.isfile(path)):
        raise FileNotFoundError(f"File not found: {path}")


    return path

def verify_files(list_file_path: str) -> Tuple[List[str], List[str]]:
    """
    Verifies the existence of files listed in a text file.

    Parameters:
        list_file_path (str): Path to a text file containing one file path per line.

    Returns:
        Tuple[List[str], List[str]]:
            A tuple containing:
                - List[str]: All existing file paths
                - List[str]: All missing file paths
    """

    existing: List[str] = []
    missing: List[str] = []

    # Read the text file containing file paths
    with open(list_file_path, "r") as f:
        for line in f:
            path: str = line.strip()
            if not path:
                continue  # skip blank lines
            path = try_fix_path(path)




            if file_exists(path, ignore_case=False):
                existing.append(path)
            else:
                missing.append(path)


    # Print results
    print(f"Total files listed: {len(existing) + len(missing)}")
    print(f"Existing files: {len(existing)}")
    print(f"Missing files: {len(missing)}\n")

    return existing, missing



def open_file_with_vs_code(path: str) -> None:
    if True:
        subprocess.run(["/mnt/c/Users/brant.buchika/AppData/Local/Programs/Microsoft VS Code/bin/code", path])
    else: #seems broken
        if sys.platform.startswith("win"):
            os.startfile(path)  # Windows
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])  # macOS
        else:
            subprocess.run(["xdg-open", path])  # Linux / WSL


from typing import Final


def generate_linux_paths_file_from_win_paths_file(winPathsFile: str) -> str:
    """
    Generates a Linux-path file from a file containing Windows paths.

    Rules:
    - Ensures the input file exists
    - Output file name replaces trailing '.txt' with '.lin.txt'
    - Replaces:
        d:  -> WIN_DRIVES_FOLDER/d
        \\  -> /
    - tries to fix common case-sensitivity issues (forces /videos/ to /Videos/, for example)
    - Writes one Linux path per line
    - Returns the path to the generated Linux paths file
    """

    # --- Ensure input file exists ---
    if not os.path.isfile(winPathsFile):
        raise FileNotFoundError(f"Input file does not exist: {winPathsFile}")

    # --- Create output filename ---
    if is_win_paths_file(winPathsFile):
        linPathsFile: Final[str] = get_lin_paths_file_from_win_paths_file(winPathsFile)
    else:
        raise ValueError(f"Input file does not appear to be a Windows paths file: {winPathsFile}")


    # --- Convert paths ---
    with open(winPathsFile, "r", encoding="utf-8") as infile, \
         open(linPathsFile, "w", encoding="utf-8") as outfile:

        for line in infile:
            win_path: str = line.strip()
            if not win_path:
                continue  # skip empty lines

            lin_path: str = win_path

            # Normalize drive letter (case-insensitive)
            lin_path = lin_path.replace("D:", f"{WIN_DRIVES_FOLDER}/d").replace("d:", f"{WIN_DRIVES_FOLDER}/d")

            # Replace backslashes with forward slashes
            lin_path = lin_path.replace("\\", "/")
            lin_path = try_fix_path(lin_path)

            outfile.write(lin_path + "\n")

    return linPathsFile

def is_linux_paths_file(winPathsFile: str) -> bool:
    return winPathsFile.lower().endswith(LIN_PATHS_SUFFIX)

def is_win_paths_file(winPathsFile: str) -> bool:
    return winPathsFile.lower().endswith(WIN_PATHS_SUFFIX) and not is_linux_paths_file(winPathsFile)

def get_win_paths_file_from_lin_paths_file(linPathsFile: str) -> str:
    if is_linux_paths_file(linPathsFile):
        return linPathsFile[:-len(LIN_PATHS_SUFFIX)] + WIN_PATHS_SUFFIX
    else:
        raise ValueError(f"Input file does not appear to be a Linux paths file: {linPathsFile}")


def get_lin_paths_file_from_win_paths_file(winPathsFile: str) -> str:
    if is_win_paths_file(winPathsFile):
        return winPathsFile[:-len(WIN_PATHS_SUFFIX)] + LIN_PATHS_SUFFIX
    else:
        raise ValueError(f"Input file does not appear to be a Windows paths file: {winPathsFile}")


# Example usage:
if __name__ == "__main__":

    # imgListPath = "/home/brant/datasets/d/Videos/DL/Training/2025/Therm20251104/valSubset2.lin.txt"
    linPathsFile = R"/mnt/d/videos/DL/Training/2025/Therm20251104/trainSubset.lin.txt"  # Input Linux paths file

    if not os.path.isfile(linPathsFile):
        winPathsFile = get_win_paths_file_from_lin_paths_file(linPathsFile)
        generate_linux_paths_file_from_win_paths_file(winPathsFile)

    existing, missing = verify_files(linPathsFile)

    logPath = "../../missing_files.txt"
    with open(logPath, "w") as out:
        for m in missing:
            out.write(m + "\n")

    if missing:
        open_file_with_vs_code(logPath)