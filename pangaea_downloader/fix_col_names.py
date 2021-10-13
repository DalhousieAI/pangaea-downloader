import os

import pandas as pd


def fix_col_names(df):
    # Lists are mutable, indexes are not
    new_cols = df.columns.copy().to_list()

    # Check each column name and map to appropriate name
    for i, col in enumerate(df.columns):
        # TODO: Replace with mappings in the BenthicNet contacts and data file
        # Match
        if col == "Dataset":
            # Replace
            new_cols[i] = "dataset_title"

    # Change columns
    df.columns = new_cols
    return df


def walk_dir(directory="."):
    print("Checking directory:", directory)
    total_files = 0
    for root_path, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root_path, file)
            df = pd.read_csv(filepath)
            if df is not None:
                total_files += 1
    print("Total files:", total_files)


if __name__ == "__main__":
    walk_dir("../test-dir")
