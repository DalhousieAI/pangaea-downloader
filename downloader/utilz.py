import requests
from bs4 import BeautifulSoup
from pandas import DataFrame, concat
from pangaeapy import PanDataSet
from requests.compat import urljoin


# -------------------------- Functions for PanQuery Scraper -------------------------- #
def has_url_col(df: DataFrame) -> bool:
    """Take a Pandas DataFrame and return True if it has a URL column."""
    return any(["url" in col.lower() for col in df.columns]) or any(
        ["image" in col.lower() for col in df.columns]
    )


def set_metadata(df: DataFrame, alt="unknown") -> DataFrame:
    """Add metadata to dataframe."""
    df.data["Dataset"] = df.title
    df.data["DOI"] = df.doi
    df.data["Campaign"] = (
        df.events[0].campaign.name if df.events[0].campaign is not None else alt
    )
    df.data["Site"] = df.data["Event"]
    return df


def fetch_child_datasets(parent: PanDataSet) -> DataFrame:
    """
    Fetch child datasets of a parent and return a merged dataset of all children.
    """
    df_list = []
    print(f"\t[INFO] Fetching {len(parent.children)} child datasets...")

    for doi in parent.children:
        # Fetch child dataset
        child = PanDataSet(doi)
        if not has_url_col(child.data):
            print(
                f"\t[WARNING] Image URL column NOT FOUND! "
                f"Data will NOT be saved! DOI: {child.doi}"
            )
        else:
            child.data = set_metadata(child.data)
            # Add child dataset to list
            df_list.append(child.data)

    # List not empty
    if len(df_list) > 0:
        print("\t[INFO] Joining child datasets...")
        df = concat(df_list, ignore_index=True)
    else:  # Empty
        df = None

    return df


def url_from_doi(doi):
    """Take a Pangaea doi string and return the url to the dataset."""
    a, b = doi.split(":")
    url = "https://" + a + ".org/" + b
    return url


# --------------- Functions for datasets with images hosted on website --------------- #
def get_metadata(page_soup):
    coordinates = page_soup.find("div", attrs={"class": "hanging geo"})
    lat = float(coordinates.find("span", attrs={"class": "latitude"}).text)
    long = float(coordinates.find("span", attrs={"class": "longitude"}).text)
    return lat, long


def get_pagination(page_soup, src_url="https://www.pangaea.de/helpers/Benthos.php"):
    """
    Take a BeautifulSoup object and return a dictionary with page numbers and URLs.
    """
    # <p> tag containing pagination info
    pagination = page_soup.find("p", attrs={"class": "navigation"})
    # Page numbers (strs)
    page_nums = [i.strip() for i in pagination.text.split("|")][2:-1]
    # List of page URLs
    page_urls = [urljoin(src_url, a["href"]) for a in pagination.find_all("a")][:-1]
    # Page number : Page URL
    page_dict = {k: v for k, v in zip(page_nums, page_urls)}
    return page_dict


def get_image_urls(page_soup, verbose=False):
    """
    Take a BeautifulSoup object and return list of image urls.
    """
    urls = []

    table = page_soup.find("table", class_="pictable")
    photos = table.find_all("td")
    if verbose:
        print("[INFO] Number of photos on page:", len(photos))

    # urls = ["https:"+td.a['href'] for td in photos]
    for td in photos:
        try:
            url = "https:" + td.a["href"]
            urls.append(url)
        except TypeError:
            # The last <td> of the last page is sometimes empty
            # No photos, just a blank <td> tag
            print("[WARNING] Empty <td> tag encountered!")

    return urls


def scrape_dataset(page_soup):
    pagination = get_pagination(page_soup)
    # Scrape current page
    print("[INFO] Processing Page 1...")
    img_urls = get_image_urls(page_soup, verbose=True)
    # Scraper subsequent pages
    for n in pagination:
        print(f"[INFO] Processing Page {n}...")
        url = pagination[n]
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "lxml")
        urls = get_image_urls(soup, verbose=True)
        img_urls.extend(urls)
    return img_urls
