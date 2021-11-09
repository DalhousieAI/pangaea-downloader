"""
Exclude datasets that are not of benthic habitat images.

- Load all csv files in input directory.
- Plot middle image of each dataset.
- After showing each image the program asks for user input.
    1. Delete dataset
    - If the image is not a seafloor photograph then the user should enter "delete" into the console.
    - This will move the file to the following path: "input-directory/.discarded/".
    2. Keep dataset
    - To keep the dataset, the user may press enter/return (or any input other than "delete").
    3. Exit program
    - The user may also exit the program while by entering "exit".
- At the end of the loop the program shows the number of datasets discarded.
"""

import os
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from pangaea_downloader.tools.checker import COMPRESSED_FILE_EXTENSIONS, is_img_url
from pangaea_downloader.tools.datasets import get_dataset_id, get_url_col
from pangaea_downloader.tools.eda import img_from_url


def get_file_paths(data_dir: str) -> List[str]:
    """Take a directory as input and return a list of paths to all files inside."""
    file_paths = []
    for root_path, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                file_paths.append(os.path.join(root_path, file))
    return file_paths


def evaluate_dataset(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Take a DataFrame, plot it's middle image and evaluate if it should be discarded."""
    filename = get_dataset_id(df) + ".csv"
    url_col = get_url_col(df)
    urls = df[url_col].dropna()
    # Plot the middle image
    idx = urls.size // 2
    sample = urls.iloc[idx]
    if not isinstance(sample, str):
        print(f"[ERROR] Non-string image URL value! {sample}")
        return
    if sample.endswith(COMPRESSED_FILE_EXTENSIONS):
        print(
            f"[WARNING] Zipped dataset might contain images."
            f" Keeping {filename}: '{sample}'"
        )
        return
    elif is_img_url(sample):
        # Load image
        print("Making get request...")
        img = img_from_url(sample, verbose=True)
        # Plot
        plt.imshow(img)
        plt.title(f"File: {filename}")
        plt.show()
    else:
        print(f"[ERROR] INVALID URL in {filename}: '{sample}'")
    # Take user input
    in_ = input("Write 'delete' to remove this dataset >> ").lower()
    return in_, filename


def exclude_datasets(data_dir: str):
    """Take a directory, load the dataset files and discard irrelevant datasets."""
    assert os.path.exists(data_dir), f"Directory does not exist: '{data_dir}'!"
    # Create folder for discarded datasets
    rem_dir = os.path.join(data_dir, ".discarded")
    os.makedirs(rem_dir, exist_ok=True)
    # Load datasets
    file_paths = get_file_paths(data_dir)
    print(f"Total {len(file_paths)} files to process.")
    # Dictionary of path: dataframe
    datasets = {path: pd.read_csv(path) for path in file_paths}
    # Sort dictionary by length of dataframes
    sorted_datasets = dict(
        sorted(datasets.items(), key=lambda item: len(item[1]), reverse=True)
    )
    # Evaluate each dataset
    discards = 0
    for i, (path, df) in enumerate(sorted_datasets.items()):
        print(f"\n[{i+1}] Processing '{path}'...")
        print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns.")
        ret = evaluate_dataset(df)
        if ret is not None:
            x, file = ret
            if x == "delete":
                move_file(path, os.path.join(rem_dir, file))
                print(f"Removed file ({len(df)} rows of data) to discards folder")
                discards += 1
            elif x == "exit":
                print("Quitting program...")
                sys.exit()
    print(f"\nCOMPLETED! Files discarded: {discards}.")


def move_file(old_path: str, new_path: str):
    """Move file from old path to new path."""
    assert os.path.exists(old_path), f"Path does not exist: {old_path}"
    os.replace(old_path, new_path)


if __name__ == "__main__":
    exclude_datasets("../query-outputs")
