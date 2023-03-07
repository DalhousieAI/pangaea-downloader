"""Functions for scraping image urls and metadata from paginated datasets."""
import time
from typing import List, Optional, Tuple

import colorama
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
from pangaeapy import PanDataSet
from requests.compat import urljoin

import pangaea_downloader.tools.datasets as datasets


def scrape_image_data(url: str, verbose=1) -> Optional[DataFrame]:
    """Scrape image URLs and metadata from webpage(s)."""
    # Load dataset
    t_wait = max(0, datasets.T_POLL_LAST + datasets.T_POLL_INTV - time.time())
    time.sleep(t_wait)  # Stay under 180 requests every 30s
    ds = PanDataSet(url)
    datasets.T_POLL_LAST = time.time()
    # Request dataset url
    if verbose >= 1:
        print("\t\t\t[INFO] Requesting:", url)
    resp = requests.get(url)
    # Parse response
    soup = BeautifulSoup(resp.text, "lxml")
    # Get coordinates of expedition
    coordinates = get_metadata(soup)

    # Get download link to photos page
    download_link = soup.find("div", attrs={"class": "text-block top-border"}).a["href"]
    src_url = download_link.split("?")[0]
    if verbose >= 1:
        print("\t\t\t[INFO] URL to photos page:", download_link)
    # Get to photos page (page 1)
    resp = requests.get(download_link)
    photos_page = BeautifulSoup(resp.text, "lxml")
    img_urls = get_urls_from_each_page(photos_page, src_url, verbose=verbose)
    if img_urls is None:
        return
    # Store URLs and add metadata
    df = DataFrame(img_urls, columns=["URL"])
    df = datasets.exclude_rows(df)
    df["Filename"] = df["URL"].apply(lambda link: link.split("/")[-1])
    if coordinates is not None:
        lat, long = coordinates
        df["Longitude"] = long
        df["Latitude"] = lat
    df["Dataset"] = ds.title
    df["DOI"] = getattr(ds, "doi", "")
    doi = getattr(ds, "doi", "").split("doi.org/")[-1]
    if (len(ds.events) > 0) and (ds.events[0].campaign is not None):
        df["Campaign"] = ds.events[0].campaign.name
    else:
        df["Campaign"] = doi
    if "Event" in ds.data.columns:
        df["Site"] = ds.data["Event"]
    else:
        df["Site"] = doi + "_site"
    return df


def get_metadata(page_soup: BeautifulSoup) -> Optional[Tuple[float, float]]:
    """Extract dataset latitude and longitude from parsed BeautifulSoup object of page."""
    assert isinstance(
        page_soup, BeautifulSoup
    ), f"invalid input type: {type(page_soup)}"
    coordinates = page_soup.find("div", attrs={"class": "hanging geo"})
    if coordinates is not None:
        lat = float(coordinates.find("span", attrs={"class": "latitude"}).text)
        long = float(coordinates.find("span", attrs={"class": "longitude"}).text)
        return lat, long
    print(
        colorama.Fore.RED + "\t\t\t[ERROR] Coordinate metadata not found on page!"
        " Saved file won't have Longitude, Latitude columns!" + colorama.Fore.RESET
    )
    return None


def get_urls_from_each_page(
    page_soup: BeautifulSoup, base_url: str, verbose=1
) -> List[str]:
    """Scrape image URLs from each page."""
    pagination = get_pagination(page_soup, base_url)
    if verbose >= 1:
        print("\t\t\t[INFO] Processing Page 1...")
    img_urls = get_page_image_urls(page_soup, verbose=verbose)
    if pagination is not None:
        for n in pagination:
            if verbose >= 1:
                print(f"\t\t\t[INFO] Processing Page {n}...")
            url = pagination[n]
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "lxml")
            urls = get_page_image_urls(soup, verbose=verbose)
            img_urls.extend(urls)
    return img_urls


def get_pagination(page_soup: BeautifulSoup, src_url: str) -> Optional[dict]:
    """
    Take a BeautifulSoup object and return a dictionary with page number and URL key, value pairs.
    """
    # <p> tag containing pagination info
    pagination = page_soup.find("p", attrs={"class": "navigation"})
    if pagination is None:
        return None
    # Page numbers (strs)
    page_nums = [i.strip() for i in pagination.text.split("|")][2:-1]
    # List of page URLs
    page_urls = [urljoin(src_url, a["href"]) for a in pagination.find_all("a")][:-1]
    # Page number : Page URL
    page_dict = {k: v for k, v in zip(page_nums, page_urls)}
    return page_dict


def get_page_image_urls(page_soup: BeautifulSoup, verbose=1) -> Optional[List[str]]:
    """Take a BeautifulSoup object and return list of image urls."""
    table = page_soup.find("table", class_="pictable")
    if table is None:
        if verbose >= 1:
            print(
                colorama.Fore.RED
                + "\t\t\t[ERROR] Image table not found: no <table> of class='pictable'!"
                + colorama.Fore.RESET
            )
        return
    photos = table.find_all("td")

    urls = []
    empty_tds = 0
    for td in photos:
        try:
            url = "https:" + td.a["href"]
            urls.append(url)
        except TypeError:
            # The last <td> of the last page is sometimes empty
            # No photos, just a blank <td> tag
            empty_tds += 1
    # Number of photos on page
    if verbose >= 2:
        n = len(photos) - empty_tds
        print(f"\t\t\t[INFO] Number of photos on page: {n}")
    return urls
