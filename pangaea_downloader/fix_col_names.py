import os

import pandas as pd


def load_column_mappings(file="mappings.xlsx", sheet_name="Data Fields"):
    # Load CSV file
    mappings_df = pd.read_excel(file, sheet_name=sheet_name)
    mappings_df.drop(["Contents", "Mandatory/Optional"], axis="columns", inplace=True)

    # Convert to dict
    maps = {}
    for _, row in mappings_df.iterrows():
        col_name, equiv_list = row[0], row[1]
        if isinstance(equiv_list, str):
            maps[col_name] = [eq.strip() for eq in equiv_list.split(",")]
    return maps


def find_col_equivalent(col_name, mapping_dict):
    # Look for equivalency and return
    for std_col in mapping_dict:
        for eq in mapping_dict[std_col]:
            if col_name.lower() == eq:
                return std_col
            elif col_name.lower() == "dataset":
                return "dataset_title"
    # If none found return original input in lower case
    return col_name.lower()


def fix_col_names(df, mapping_dict, verbose=False):
    if not isinstance(df, pd.DataFrame):
        return
    # Lists are mutable, indexes are not
    new_cols = df.columns.copy().to_list()

    # Check each column name and map to appropriate name
    for i, col in enumerate(df.columns):
        new_cols[i] = find_col_equivalent(col, mapping_dict)

    if verbose:
        print("Original columns:", df.columns.to_list())
        print("After modifying :", new_cols)

    # Change columns
    df.columns = new_cols
    return df


def walk_dir(directory="."):
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"The directory '{directory}' doesn't exists")
    print("Checking directory:", directory)
    maps = load_column_mappings()
    total_files = 0
    for root_path, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root_path, file)
            df = pd.read_csv(filepath) if filepath.endswith(".csv") else None
            df = fix_col_names(df, maps)
            if df is not None:
                df.to_csv(filepath, index=False)
                total_files += 1
    print("Total files:", total_files)


if __name__ == "__main__":
    walk_dir("../test-dir")
