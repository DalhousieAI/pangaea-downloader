from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from pandas import DataFrame, concat
from pangaeapy import PanDataSet, PanQuery
from requests.compat import urljoin


#  --------- Basic functions for searching and parsing results --------- #
def search_pangaea(query: str, n_results: int) -> List[dict]:
    """Search Pangaea with given query string and return a list of results."""
    offset = 0
    results = []
    while True:
        pq = PanQuery(query=query, limit=n_results, offset=offset)
        results.extend(pq.result)
        offset += len(pq.result)
        if len(results) >= pq.totalcount:
            break
    # Sanity check
    assert len(results) == pq.totalcount
    print(f"[INFO] Number of results returned: {len(results)}")
    return results


def get_result_info(result: dict) -> Tuple[str, str, str, bool]:
    """
    Process result item and returns the dataset url, size and if it is a parent.
    """
    # Parse html
    soup = BeautifulSoup(result["html"], "lxml")
    # Extract info
    citation = soup.find("div", attrs={"class": "citation"}).text
    url = soup.find("a", attrs={"class": "dataset-link"})["href"]
    size = soup.find_all("td", class_="content")[-1].text.lower()
    is_parent = True if result["type"] == "parent" else False
    return citation, url, size, is_parent


# -------------- Functions for handling each dataset type -------------- #
# # Functions for Type: Paginated
def scrape_images(url: str) -> DataFrame:
    """Scrape image URLs and metadata from webpage(s)."""
    # Load dataset
    ds = PanDataSet(url)
    # Request dataset url
    print("\t[INFO] Requesting:", url)
    resp = requests.get(url)
    # Parse response
    soup = BeautifulSoup(resp.text, "lxml")
    # Get coordinates of expedition
    lat, long = get_metadata(soup)

    # Get download link to photos page
    download_link = soup.find("div", attrs={"class": "text-block top-border"}).a["href"]
    print("\t[INFO] URL to photos page:", download_link)
    # Get to photos page (page 1)
    resp = requests.get(download_link)
    photos_page = BeautifulSoup(resp.text, "lxml")
    img_urls = scrape_dataset(photos_page)

    # Store URLs and add metadata
    df = DataFrame(img_urls, columns=["URL"])
    df["Filename"] = df["URL"].apply(lambda link: link.split("/")[-1])
    df["Longitude"] = long
    df["Latitude"] = lat
    df["Site"] = ds.events[0].label
    df["Campaign"] = ds.events[0].campaign.name
    df["Dataset"] = ds.title
    return df


def get_metadata(page_soup: BeautifulSoup) -> Tuple[float, float]:
    """Extract dataset latitude and longitude from parsed BeautifulSoup object of page."""
    coordinates = page_soup.find("div", attrs={"class": "hanging geo"})
    lat = float(coordinates.find("span", attrs={"class": "latitude"}).text)
    long = float(coordinates.find("span", attrs={"class": "longitude"}).text)
    return lat, long


def get_pagination(
    page_soup: BeautifulSoup, src_url="https://www.pangaea.de/helpers/Benthos.php"
) -> Optional[dict]:
    """
    Take a BeautifulSoup object and return a dictionary with page number and URL key, value pairs.
    """
    # <p> tag containing pagination info
    pagination = page_soup.find("p", attrs={"class": "navigation"})
    if pagination is None:
        return None
    else:
        # Page numbers (strs)
        page_nums = [i.strip() for i in pagination.text.split("|")][2:-1]
        # List of page URLs
        page_urls = [urljoin(src_url, a["href"]) for a in pagination.find_all("a")][:-1]
        # Page number : Page URL
        page_dict = {k: v for k, v in zip(page_nums, page_urls)}
        return page_dict


def get_image_urls(page_soup: BeautifulSoup, verbose=False) -> List[str]:
    """Take a BeautifulSoup object and return list of image urls."""
    table = page_soup.find("table", class_="pictable")
    photos = table.find_all("td")
    print("\t[INFO] Number of photos on page:", len(photos)) if verbose else 0

    urls = []
    for td in photos:
        try:
            url = "https:" + td.a["href"]
            urls.append(url)
        except TypeError:
            # The last <td> of the last page is sometimes empty
            # No photos, just a blank <td> tag
            print("\t[WARNING] Empty <td> tag encountered!")

    return urls


def scrape_dataset(page_soup: BeautifulSoup) -> List[str]:
    """Scrape image URLs from each page."""
    pagination = get_pagination(page_soup)
    print("\t[INFO] Processing Page 1...")
    img_urls = get_image_urls(page_soup, verbose=True)
    if pagination is not None:
        for n in pagination:
            print(f"\t[INFO] Processing Page {n}...")
            url = pagination[n]
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "lxml")
            urls = get_image_urls(soup, verbose=True)
            img_urls.extend(urls)
    return img_urls


# # Functions for Type: Parent
def fetch_child_datasets(url: str) -> Optional[DataFrame]:
    ds = PanDataSet(url)
    ds_id = ds.doi.split("PANGAEA.")[-1]
    print(f"\t[INFO] Fetching {len(ds.children)} child datasets...")
    # Dataset is restricted
    if ds.loginstatus != "unrestricted":
        print(f"\t[ERROR] Access restricted: '{ds.loginstatus}'. URL: {url}")
        return

    df_list = []
    for i, doi in enumerate(ds.children):
        # Load child dataset
        child = PanDataSet(doi)
        # Check for image URL column
        if not has_url_col(child.data):
            print(
                f"\t\t[{i+1}] [WARNING] Image URL columns NOT found! DOI: {child.doi} Skipping..."
            )
        else:
            # Add metadata
            child.data = set_metadata(child, alt=ds_id)
            # Add child dataset to list
            df_list.append(child.data)

    if len(df_list) <= 0:
        # Empty list
        print("\t[ERROR] No child dataset had image URL column!")
        return None
    else:
        # List NOT empty
        print("\t[INFO] Joining child datasets...")
        return concat(df_list, ignore_index=True)


def has_url_col(df: DataFrame) -> bool:
    """Take a Pandas DataFrame and return True if it has image URL column."""
    return any(["url" in col.lower() for col in df.columns]) or any(
        ["image" in col.lower() for col in df.columns]
    )


def get_url_cols(df: DataFrame) -> List[str]:
    """Take a Pandas DataFrame and return a list of URL columns."""
    return [col for col in df.columns if ("url" in col.lower())]


def set_metadata(ds: PanDataSet, alt="unknown") -> DataFrame:
    """Add metadata to dataframe."""
    ds.data["Dataset"] = ds.title
    ds.data["DOI"] = ds.doi
    ds.data["Campaign"] = (
        ds.events[0].campaign.name if ds.events[0].campaign is not None else alt
    )
    ds.data["Site"] = ds.data["Event"]
    return ds.data


# # Functions for Type: Tabular
def fetch_dataset(url: str) -> Optional[DataFrame]:
    """Fetch Pangaea dataset using provided URI/DOI and return DataFrame."""
    # Load data set
    ds = PanDataSet(url)
    # Dataset is restricted
    if ds.loginstatus != "unrestricted":
        print(f"\t[ERROR] Access restricted: '{ds.loginstatus}'. URL: {url}")
        return
    # Check for image URL column
    if not has_url_col(ds.data):
        print("\t[WARNING] Image URL columns NOT found! Skipping...")
        return
    # Add metadata
    ds.data = set_metadata(ds, alt=ds.doi)
    return ds.data
