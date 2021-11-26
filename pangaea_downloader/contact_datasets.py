import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pangaea_downloader.exclude_datasets import get_file_paths
from pangaea_downloader.fix_col_names import load_column_mappings
from pangaea_downloader.tools.checker import is_url


class PangaeaBenthicImageDataset:
    def __init__(self):
        """Initialize Pangaea Benthic Image Dataset class."""
        self.mappings_file = "mappings.xlsx"
        # List of standard column names
        self.std_cols = list(load_column_mappings(file=self.mappings_file).keys())
        if "dataset_title" not in self.std_cols:
            self.std_cols.insert(0, "dataset_title")
        # Initialize dataframe
        self.full_df = pd.DataFrame(columns=self.std_cols)

    def concat_dfs(self, data_dir="../query-outputs/", discards_dir=".discarded"):
        """Load each dataset and concatenate them."""
        n_files = 0
        file_paths = get_file_paths(data_dir)
        # Load and concatenate each dataset
        for f_path in file_paths:
            directory, file_name = os.path.split(f_path)
            # Exclude files in the discards directory
            if not directory.endswith(discards_dir):
                n_files += 1
                print(f"[{n_files}] Loading '{f_path}'")
                df = pd.read_csv(f_path)
                self.full_df = pd.concat([self.full_df, df])
                self.full_df = self.full_df.reset_index(drop=True)
                # Make sure only standard columns are kept
                if not sorted(self.full_df) == sorted(self.std_cols):
                    self.full_df = self.full_df[self.std_cols]
        print(f"Total {n_files} datasets concatenated.")
        print(
            "DataFrame has only standard columns:",
            sorted(self.full_df) == sorted(self.std_cols),
        )

    def save(self, save_dir="../full_dataset", file_name="pangaea-dataset.csv"):
        """Save dataset in the specified directory with the specified file name."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, file_name)
        if True:
            self.full_df.to_csv(path, index=False)
            print("Full dataset saved to", path)

    def is_null(self):
        """Show missing values plot and counts."""
        # Missing values plot
        plt.figure(figsize=(16, 4))
        sns.heatmap(self.full_df.isna(), cmap="viridis")
        plt.show()
        # Missing values counts
        mapping_df = pd.read_excel(self.mappings_file, index_col=0)
        counts = self.full_df.count()
        total = self.full_df.isna().sum()
        perc = (total / len(self.full_df) * 100).round(3)
        missing = pd.DataFrame(
            {
                "counts": counts,
                "missing": total,
                "missing_percent": perc,
                "col_required": mapping_df["Mandatory/Optional"],
            }
        )
        missing["_"] = "%"
        missing["col_required"].fillna("Optional", inplace=True)
        missing.sort_values(by="col_required")
        print(missing)

    def frequent_url_ext(self, top_n=40):
        """Show top_n number of most frequent URL file extensions in dataset."""
        # Analyze URLs and file extensions
        urls = self.full_df.url.dropna()
        valid_urls = urls[urls.apply(is_url)]
        ext_list = []
        for url in valid_urls:
            ext = url.split(".")[-1].lower()
            if len(ext) < 5:
                ext_list.append(ext)
        print(f"Showing top {top_n} most frequent file extensions")
        print(pd.Series(ext_list).value_counts().iloc[:top_n])


if __name__ == "__main__":
    pangaea_ds = PangaeaBenthicImageDataset()
    pangaea_ds.concat_dfs(data_dir="../query-outputs-stable/")
    pangaea_ds.save(save_dir="../full_dataset")
