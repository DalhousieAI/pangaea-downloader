"""
Functions for fetching Pangaea Datasets.

Note: this module is only for Parent and Child datasets.
      For paginated datasets (images hosted on webpages)
      use pangaea_downloader.tools.scraper module.
"""
import os
from typing import Optional

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
        print("\t[WARNING] Image URL column NOT found! Skipping...")
        return
    # Add metadata
    df = set_metadata(ds, alt=doi)
    return df


def fetch_children(parent_url: str, out_dir: str):
    """Take in url of a parent dataset and output directory path, fetch and save child datasets to file."""
    # Fetch dataset
    ds = PanDataSet(parent_url)
    # Check restriction
    if ds.loginstatus != "unrestricted":
        print(f"\t[ERROR] Access restricted: '{ds.loginstatus}'. URL: {parent_url}")
        return
    # Process children
    print(f"\t[INFO] Fetching {len(ds.children)} child datasets...")
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
            child = PanDataSet(url)
            # Save scrapped dataset
            save_df(df, child.id, out_dir)
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
                # Save child dataset
                save_df(df, child.id, out_dir)


def set_metadata(ds: PanDataSet, alt="unknown") -> DataFrame:
    """Add metadata to a PanDataSet's dataframe."""
    ds.data["Dataset"] = ds.title
    ds.data["DOI"] = ds.doi
    # Dataset campaign
    if (len(ds.events) > 0) and (ds.events[0].campaign is not None):
        ds.data["Campaign"] = ds.events[0].campaign.name
    else:
        ds.data["Campaign"] = alt
    # Dataset site/event/deployment
    if "Event" in ds.data.columns:
        ds.data["Site"] = ds.data["Event"]
    else:
        ds.data["Site"] = alt + "_site"
    return ds.data


def save_df(df: DataFrame, ds_id: str, output_dir: str):
    """Save a DataFrame to file in a provided output directory."""
    f_name = ds_id + ".csv"
    path = os.path.join(output_dir, f_name)
    df.to_csv(path, index=False)
    print(f"\t\t[INFO] Saved to '{path}'")
