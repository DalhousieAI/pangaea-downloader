import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from pangaea_downloader.tools.checker import is_url
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


def evaluate_dataset(df: pd.DataFrame) -> Tuple[str, str]:
    """Take a DataFrame, plot it's middle image and evaluate if it should be discarded."""
    filename = get_dataset_id(df) + ".csv"
    url_col = get_url_col(df)
    urls = df[url_col].dropna()
    # Plot the middle image
    idx = urls.size // 2
    sample = urls.iloc[idx]
    if is_url(sample):
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


if __name__ == "__main__":
    print(get_file_paths("../test-dir"))
