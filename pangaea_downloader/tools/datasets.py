"""
Functions for fetching Pangaea Datasets.

Note: this module is only for Parent and Child datasets.
      For paginated datasets (images hosted on webpages)
      use pangaea_downloader.tools.scraper module.
"""
import os
from typing import List, Optional

from pandas import DataFrame
from pangaeapy import PanDataSet

from pangaea_downloader.tools import checker, process, scraper


def fetch_child(child_url: str) -> Optional[DataFrame]:
    """Fetch Pangaea child dataset using provided URI/DOI and return DataFrame."""
    # Load data set
    ds = PanDataSet(child_url)
    doi = ds.doi.split("doi.org/")[-1]
    # Dataset is restricted
    if ds.loginstatus != "unrestricted":
        print(f"\t[ERROR] Access restricted: '{ds.loginstatus}'. URL: {child_url}")
        return
    # Check for image URL column
    if not checker.has_url_col(ds.data):
        print(
            f"\t[WARNING] Image URL column NOT found! Columns: {list(ds.data.columns)}. Skipping..."
        )
        return
    # Add metadata
    df = set_metadata(ds, alt=doi)
    # Exclude unwanted rows
    df = exclude_rows(df)
    return df


def fetch_children(parent_url: str) -> Optional[List[DataFrame]]:
    """Take in url of a parent dataset, fetch and return list of child datasets."""
    # Fetch dataset
    ds = PanDataSet(parent_url)
    # Check restriction
    if ds.loginstatus != "unrestricted":
        print(f"\t[ERROR] Access restricted: '{ds.loginstatus}'. URL: {parent_url}")
        return
    # Process children
    print(f"\t[INFO] Fetching {len(ds.children)} child datasets...")
    df_list = []
    for i, child_uri in enumerate(ds.children):
        url = process.url_from_uri(child_uri)
        size = process.get_html_info(url)
        # Assess type
        typ = process.ds_type(size)
        if typ == "video":
            print(f"\t\t[{i+1}] [WARNING] Video dataset! {url} Skipping...")
            continue
        elif typ == "paginated":
            print(f"\t\t[{i+1}] Scrapping dataset...")
            df = scraper.scrape_image_data(url)
            if df is not None:
                df_list.append(df)
        elif typ == "tabular":
            child = PanDataSet(url)
            if ds.loginstatus != "unrestricted":
                print(
                    f"\t\t[{i+1}] [ERROR] Access restricted: '{ds.loginstatus}'. {url}"
                )
                return
            if not checker.has_url_col(child.data):
                print(
                    f"\t\t[{i+1}] [WARNING] Image URL column NOT found! {url} Skipping..."
                )
            else:
                # Add metadata
                child_doi = child.doi.split("doi.org/")[-1]
                df = set_metadata(child, alt=child_doi)
                # Add child dataset to list
                df = exclude_rows(df)
                df_list.append(df)

    # Return result
    if len(df_list) > 0:
        return df_list
    else:
        # Empty list
        return None


def set_metadata(ds: PanDataSet, alt="unknown") -> DataFrame:
    """Add metadata to a PanDataSet's dataframe."""
    ds.data["dataset_title"] = ds.title
    ds.data["doi"] = ds.doi
    # Dataset campaign
    if (len(ds.events) > 0) and (ds.events[0].campaign is not None):
        ds.data["campaign"] = ds.events[0].campaign.name
    else:
        ds.data["campaign"] = alt
    # Dataset site/event/deployment
    if "Event" in ds.data.columns:
        ds.data["site"] = ds.data["Event"]
    else:
        ds.data["site"] = alt + "_site"
    return ds.data


def save_df(df: DataFrame, output_path: str, level=1, index=None) -> bool:
    """
    Save a DataFrame to a file in the provided output directory.

    Returns False if dataframe is empty, else returns True.
    """
    # Print formatting
    tabs = "\t" * level
    idx = "INFO" if index is None else index
    # Don't save empty dataframe
    if len(df) == 0:
        print(f"{tabs}[{idx}] Empty DataFrame! File not saved!")
        return False
    # Save if dataframe not empty
    df.to_csv(output_path, index=False)
    print(f"{tabs}[{idx}] Saved to '{output_path}'")
    return True


def get_url_col(df: DataFrame) -> str:
    """Take a Pandas DataFrame and return the first URL column."""
    cols = [
        col for col in df.columns if ("url" in col.lower()) or ("image" in col.lower())
    ]
    col = cols[0] if len(cols) > 0 else None
    return col


def find_column_match(df: DataFrame, column: str) -> str:
    """
    Find a matching column name, changed by casing or non-alphanumeric characters.
    """
    if column in df.columns:
        return column
    for c in df.columns:
        if c.lower().strip(" _-/\\") == column:
            return c
    else:
        raise ValueError(
            f"No column matching {column} in dataframe with columns: {df.columns}"
        )


def exclude_rows(df: DataFrame) -> DataFrame:
    """Remove rows with unwanted file extensions and return the resulting dataframe."""
    url_col = get_url_col(df)
    if url_col is not None:
        valid_rows = ~df[url_col].apply(checker.is_invalid_file_ext)
        return df[valid_rows]
    return df


def fix_text(text: str) -> str:
    """
    Replace back slash or forward slash characters in string with underscore.

    Returns modified string.
    """
    text = text.replace("\\", "_")
    text = text.replace("/", "_")
    return text


def get_dataset_id(df: DataFrame) -> str:
    """Take a Pandas DataFrame as input and return the datasets Pangaea ID."""
    col = find_column_match("doi")
    return df[col].iloc[0].split(".")[-1]
