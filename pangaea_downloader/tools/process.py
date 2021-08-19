"""Functions for processing each of the result items."""
from typing import Tuple

import requests
from bs4 import BeautifulSoup


def url_from_uri(uri: str, base_url="https://doi.pangaea.de/") -> str:
    """Take a pangaea uri/doi string as input and return its corresponding url string."""
    url = base_url + uri.split(":")[-1]
    return url


def get_result_info(res: dict) -> Tuple[str, str, str, str, bool]:
    """
    Process result dictionary and return the dataset's citation, url, id, size and is_parent.
    """
    # Parse html
    soup = BeautifulSoup(res["html"], "lxml")
    # Extract info
    citation = soup.find("div", attrs={"class": "citation"}).text
    url = soup.find("a", attrs={"class": "dataset-link"})["href"]
    size = soup.find_all("td", class_="content")[-1].text.lower()
    is_parent = True if res["type"] == "parent" else False
    ds_id = res["URI"].split("PANGAEA.")[-1]
    return citation, url, ds_id, size, is_parent


def get_html_info(url: str):
    # Make get request to webpage
    resp = requests.get(url)
    if resp.status_code == 200:
        # Parse html
        soup = BeautifulSoup(resp.text, "lxml")
        # Extract info
        size = soup.find_all("div", attrs={"class": "descr"})[-1].text.strip().lower()
        return size


def ds_type(size: str):
    """
    Take in the size description of the dataset and return the it's type.

    possible dataset types: { 'video', "paginated", "tabular", "parent" }
    """
    size = size.lower()
    if "bytes" in size:
        return "video"
    elif "unknown" == size:
        return "paginated"
    elif "data points" in size:
        return "tabular"
    elif "datasets" in size:
        return "parent"
    else:
        raise ValueError(f"Input: {size} not understood!")
