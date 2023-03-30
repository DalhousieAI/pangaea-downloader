"""
Functions for fetching Pangaea Datasets.

Note: this module is only for Parent and Child datasets.
      For paginated datasets (images hosted on webpages)
      use pangaea_downloader.tools.scraper module.
"""
import os
import time
from typing import List, Optional

import colorama
from pandas import DataFrame
from pangaeapy import PanDataSet

from pangaea_downloader.tools import checker, process, scraper

T_POLL_LAST = 0
T_POLL_INTV = 0.1667


def fetch_child(
    child_url: str,
    verbose=1,
    ensure_url=True,
    auth_token=None,
) -> Optional[DataFrame]:
    """Fetch Pangaea child dataset using provided URI/DOI and return DataFrame."""
    # Load data set
    global T_POLL_LAST
    global T_POLL_INTV
    t_wait = max(0, T_POLL_LAST + T_POLL_INTV - time.time())
    time.sleep(t_wait)  # Stay under 180 requests every 30s
    ds = PanDataSet(child_url, enable_cache=True, auth_token=auth_token)
    T_POLL_LAST = time.time()
    # Dataset is restricted
    if ds.loginstatus != "unrestricted":
        if verbose >= 1:
            print(
                colorama.Fore.YELLOW
                + f"\t[ERROR] Access restricted: '{ds.loginstatus}'. URL: {child_url}"
                + colorama.Fore.RESET
            )
        return
    # Check for image URL column
    if ensure_url and not checker.has_url_col(ds.data):
        if verbose >= 1:
            print(
                colorama.Fore.YELLOW
                + f"\t[WARNING] Image URL column NOT found! Columns: {list(ds.data.columns)}. Skipping..."
                + colorama.Fore.RESET
            )
        return
    # Add metadata
    df = set_metadata(ds)
    # Exclude unwanted rows
    df = exclude_rows(df)
    return df


def fetch_children(
    parent_url: str,
    verbose=1,
    ensure_url=True,
    auth_token=None,
) -> Optional[List[DataFrame]]:
    """Take in url of a parent dataset, fetch and return list of child datasets."""
    # Fetch dataset
    global T_POLL_LAST
    global T_POLL_INTV
    t_wait = max(0, T_POLL_LAST + T_POLL_INTV - time.time())
    time.sleep(t_wait)  # Stay under 180 requests every 30s
    ds = PanDataSet(parent_url, enable_cache=True, auth_token=auth_token)
    T_POLL_LAST = time.time()
    # Check restriction
    if ds.loginstatus != "unrestricted":
        if verbose >= 1:
            print(
                colorama.Fore.YELLOW
                + f"\t[ERROR] Access restricted: '{ds.loginstatus}'. URL: {parent_url}"
                + colorama.Fore.RESET
            )
        return
    # Process children
    if verbose >= 1:
        print(f"\t[INFO] Fetching {len(ds.children)} child datasets...")
    df_list = []
    for i, child_uri in enumerate(ds.children):
        url = process.url_from_uri(child_uri)
        size = process.get_html_info(url)
        # Assess type
        try:
            typ = process.ds_type(size)
        except Exception:
            raise ValueError(f"Can't process type from size for {url}")
        if typ == "video":
            if verbose >= 1:
                print(
                    colorama.Fore.YELLOW
                    + f"\t\t[{i+1}] [WARNING] Video dataset! {url} Skipping..."
                    + colorama.Fore.RESET
                )
            continue
        elif typ == "paginated":
            if verbose >= 1:
                print(f"\t\t[{i+1}] Scraping dataset...")
            df = scraper.scrape_image_data(url)
            if df is not None:
                df_list.append(df)
        elif typ == "tabular":
            t_wait = max(0, T_POLL_LAST + T_POLL_INTV - time.time())
            time.sleep(t_wait)  # Stay under 180 requests every 30s
            child = PanDataSet(url, enable_cache=True, auth_token=auth_token)
            T_POLL_LAST = time.time()
            if ds.loginstatus != "unrestricted":
                if verbose >= 1:
                    print(
                        colorama.Fore.YELLOW
                        + f"\t\t[{i+1}] [ERROR] Access restricted: '{ds.loginstatus}'. {url}"
                        + colorama.Fore.RESET
                    )
                continue
            if ensure_url and not checker.has_url_col(child.data):
                if verbose >= 1:
                    print(
                        colorama.Fore.YELLOW
                        + f"\t\t[{i+1}] [WARNING] Image URL column NOT found! {url} Skipping..."
                        + colorama.Fore.RESET
                    )
            else:
                # Add metadata
                df = set_metadata(child)
                # Add child dataset to list
                df = exclude_rows(df)
                df_list.append(df)

    # Return result
    if len(df_list) > 0:
        return df_list
    else:
        # Empty list
        return None


def set_metadata(ds: PanDataSet) -> DataFrame:
    """Add metadata to a PanDataSet's dataframe."""
    ds.data["dataset_title"] = ds.title
    ds.data["doi"] = getattr(ds, "doi", "")
    # Dataset campaign
    if (len(ds.events) > 0) and (ds.events[0].campaign is not None):
        ds.data["campaign"] = ds.events[0].campaign.name
    return ds.data


def save_df(df: DataFrame, output_path: str, level=1, index=None, verbose=1) -> bool:
    """
    Save a DataFrame to a file in the provided output directory.

    Returns False if dataframe is empty, else returns True.
    """
    # Print formatting
    tabs = "\t" * level
    idx = "INFO" if index is None else index
    # Don't save empty dataframe
    if len(df) == 0:
        if verbose >= 1:
            print(f"{tabs}[{idx}] Empty DataFrame! File not saved!")
        return False
    # Save if dataframe not empty
    df.to_csv(output_path, index=False)
    if verbose >= 1:
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
