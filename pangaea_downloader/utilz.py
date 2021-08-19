from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from pandas import DataFrame, concat
from pangaeapy import PanDataSet, PanQuery
from requests.compat import urljoin

from pangaea_downloader.tools import checker


#  --------- Basic functions for searching and parsing results --------- #
def run_search_query(query: str, verbose=False, n_results=500) -> List[dict]:
    """Search Pangaea with given query string and return a list of results."""
    print(f"[INFO] Running search with query string: '{query}'") if verbose else 0
    offset = 0
    results = []
    # Iteratively retrieve search results
    while True:
        pq = PanQuery(query=query, limit=n_results, offset=offset)
        results.extend(pq.result)
        offset += len(pq.result)
        if len(results) >= pq.totalcount:
            break
    # Sanity check
    assert len(results) == pq.totalcount
    print(f"[INFO] Number of search results returned: {len(results)}") if verbose else 0
    return results


def read_query_list(file="../pangaea_downloader/query_list") -> List[str]:
    """Read file with list of search queries and return it as a list."""
    with open(file, "r") as f:
        query_list = f.readlines()
    query_list = [query.strip() for query in query_list if query.strip() != ""]
    return query_list


def search_pangaea(verbose=False) -> List[dict]:
    """Search Pangaea with multiple search queries and return a list of unique results."""
    # Read in list of search queries
    query_list = read_query_list()
    # Search multiple queries
    print("[INFO] Running multiple search queries...") if verbose else 0
    results_list = []
    for i, query in enumerate(query_list):
        search_results = run_search_query(query=query, n_results=500)
        if verbose:
            print(
                f"\t[{i+1}] query: '{query}', results returned: {len(search_results)}"
            )
        results_list.extend(search_results)
    # Keep only unique results
    results_set = list({value["URI"]: value for value in results_list}.values())
    if verbose:
        print(f"[INFO] Number of unique search results: {len(results_set)}")
    return results_set


def get_result_info(result: dict) -> Tuple[str, str, str, bool]:
    """
    Process result item and returns the dataset citation, url, size and if it is a parent.
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
    src_url = download_link.split("?")[0]
    print("\t[INFO] URL to photos page:", download_link)
    # Get to photos page (page 1)
    resp = requests.get(download_link)
    photos_page = BeautifulSoup(resp.text, "lxml")
    img_urls = scrape_urls_from_each_page(photos_page, src_url)

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


def get_pagination(page_soup: BeautifulSoup, src_url: str) -> Optional[dict]:
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


def get_page_image_urls(page_soup: BeautifulSoup, verbose=False) -> List[str]:
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


def scrape_urls_from_each_page(page_soup: BeautifulSoup, base_url: str) -> List[str]:
    """Scrape image URLs from each page."""
    pagination = get_pagination(page_soup, base_url)
    print("\t[INFO] Processing Page 1...")
    img_urls = get_page_image_urls(page_soup, verbose=True)
    if pagination is not None:
        for n in pagination:
            print(f"\t[INFO] Processing Page {n}...")
            url = pagination[n]
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "lxml")
            urls = get_page_image_urls(soup, verbose=True)
            img_urls.extend(urls)
    return img_urls


# # Functions for Type: Parent
def fetch_child_datasets(url: str) -> Optional[DataFrame]:
    ds = PanDataSet(url)
    parent_doi = ds.doi.split("doi.org/")[-1]
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
        if not checker.has_url_col(child.data):
            print(
                f"\t\t[{i + 1}] [WARNING] Image URL column NOT found! DOI: {child.doi} Skipping..."
            )
        else:
            # Add metadata
            child.data = set_metadata(child, alt=parent_doi)
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


def get_url_cols(df: DataFrame) -> List[str]:
    """Take a Pandas DataFrame and return a list of URL columns."""
    return [
        col
        for col in df.columns
        if (("url" in col.lower()) or ("image" in col.lower()))
    ]


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


# # Functions for Type: Tabular
def fetch_dataset(url: str) -> Optional[DataFrame]:
    """Fetch Pangaea dataset using provided URI/DOI and return DataFrame."""
    # Load data set
    ds = PanDataSet(url)
    doi = ds.doi.split("doi.org/")[-1]
    # Dataset is restricted
    if ds.loginstatus != "unrestricted":
        print(f"\t[ERROR] Access restricted: '{ds.loginstatus}'. URL: {url}")
        return
    # Check for image URL column
    if not checker.has_url_col(ds.data):
        print("\t[WARNING] Image URL column NOT found! Skipping...")
        return
    # Add metadata
    ds.data = set_metadata(ds, alt=doi)
    return ds.data
