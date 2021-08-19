"""Functions for scraping image urls and metadata from paginated datasets."""
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
from pangaeapy import PanDataSet
from requests.compat import urljoin


def scrape_image_data(url: str) -> DataFrame:
    """Scrape image URLs and metadata from webpage(s)."""
    # Load dataset
    ds = PanDataSet(url)
    # Request dataset url
    print("\t\t\t[INFO] Requesting:", url)
    resp = requests.get(url)
    # Parse response
    soup = BeautifulSoup(resp.text, "lxml")
    # Get coordinates of expedition
    lat, long = get_metadata(soup)

    # Get download link to photos page
    download_link = soup.find("div", attrs={"class": "text-block top-border"}).a["href"]
    src_url = download_link.split("?")[0]
    print("\t\t\t[INFO] URL to photos page:", download_link)
    # Get to photos page (page 1)
    resp = requests.get(download_link)
    photos_page = BeautifulSoup(resp.text, "lxml")
    img_urls = get_urls_from_each_page(photos_page, src_url)

    # Store URLs and add metadata
    df = DataFrame(img_urls, columns=["URL"])
    df["Filename"] = df["URL"].apply(lambda link: link.split("/")[-1])
    df["Longitude"] = long
    df["Latitude"] = lat
    df["Dataset"] = ds.title
    df["DOI"] = ds.doi
    doi = ds.doi.split("doi.org/")[-1]
    if (len(ds.events) > 0) and (ds.events[0].campaign is not None):
        ds.data["Campaign"] = ds.events[0].campaign.name
    else:
        ds.data["Campaign"] = doi
    if "Event" in ds.data.columns:
        ds.data["Site"] = ds.data["Event"]
    else:
        ds.data["Site"] = doi + "_site"
    return df


def get_metadata(page_soup: BeautifulSoup) -> Tuple[float, float]:
    """Extract dataset latitude and longitude from parsed BeautifulSoup object of page."""
    coordinates = page_soup.find("div", attrs={"class": "hanging geo"})
    lat = float(coordinates.find("span", attrs={"class": "latitude"}).text)
    long = float(coordinates.find("span", attrs={"class": "longitude"}).text)
    return lat, long


def get_urls_from_each_page(page_soup: BeautifulSoup, base_url: str) -> List[str]:
    """Scrape image URLs from each page."""
    pagination = get_pagination(page_soup, base_url)
    print("\t\t\t[INFO] Processing Page 1...")
    img_urls = get_page_image_urls(page_soup, verbose=True)
    if pagination is not None:
        for n in pagination:
            print(f"\t\t\t[INFO] Processing Page {n}...")
            url = pagination[n]
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "lxml")
            urls = get_page_image_urls(soup, verbose=True)
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
    else:
        # Page numbers (strs)
        page_nums = [i.strip() for i in pagination.text.split("|")][2:-1]
        # List of page URLs
        page_urls = [urljoin(src_url, a["href"]) for a in pagination.find_all("a")][:-1]
        # Page number : Page URL
        page_dict = {k: v for k, v in zip(page_nums, page_urls)}
        return page_dict


def get_page_image_urls(page_soup: BeautifulSoup, verbose=False) -> Optional[List[str]]:
    """Take a BeautifulSoup object and return list of image urls."""
    table = page_soup.find("table", class_="pictable")
    if table is None:
        print("[ERROR] Image table not found: no <table> of class='pictable'!")
        return
    photos = table.find_all("td")
    print("\t\t\t[INFO] Number of photos on page:", len(photos)) if verbose else 0

    urls = []
    for td in photos:
        try:
            url = "https:" + td.a["href"]
            urls.append(url)
        except TypeError:
            # The last <td> of the last page is sometimes empty
            # No photos, just a blank <td> tag
            print("\t\t\t[WARNING] Empty <td> tag encountered!")

    return urls
