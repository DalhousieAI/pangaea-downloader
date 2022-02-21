"""
Scrape license information for Pangaea datasets in BenthicNet.
"""
import datetime
from typing import Dict, Optional, Union

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_dataset_url(ds_id: Union[str, int]) -> str:
    """Return dataset URL given the six digit dataset ID."""
    return f"https://doi.pangaea.de/10.1594/PANGAEA.{ds_id}"


def get_license_info(url: str) -> Optional[Dict[str, str]]:
    """Return a dictionary with license information given the dataset URL."""
    # Make a request to the URL and parse the html
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "lxml")
    # Get the tag containing the license info
    license_tag = soup.find("a", attrs={"rel": "license"})
    if license_tag is None:
        return
    return {"url": license_tag["href"], "text": license_tag.text}


def main(pangaea_file):
    # Load list of dataset IDs
    df = pd.read_csv(pangaea_file, low_memory=False)
    ds_ids = [ds_name.split("-")[-1] for ds_name in df.dataset.unique()]
    print(f"Total {len(ds_ids)} dataset licenses to fetch.")

    # Scrape license information for each dataset
    license_list = []
    for ds_id in tqdm(ds_ids):
        ds_url = get_dataset_url(ds_id)
        info = get_license_info(ds_url)
        if info is None:
            info = {"url": None, "text": None}
        info["id"] = "pangaea-" + ds_id
        license_list.append(info)

    # Save license info to file
    license_df = pd.DataFrame(license_list)
    license_df.to_csv(
        f"../pangaea-dataset-licenses-{datetime.date.today()}.csv", index=False
    )
    print(f"License information scrapped for {len(license_df)} datasets.")


if __name__ == "__main__":
    main(pangaea_file="../pangaea_2022-01-27.csv")
