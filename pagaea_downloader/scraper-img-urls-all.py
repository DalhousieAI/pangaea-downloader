import pandas as pd
import pangaeapy
import requests
from bs4 import BeautifulSoup
from utilz import get_metadata, scrape_dataset

# Christiansen, B (2006)
dois = [
    "https://doi.org/10.1594/PANGAEA.371062",
    "https://doi.org/10.1594/PANGAEA.371063",
    "https://doi.org/10.1594/PANGAEA.371064",
]


def main(out_dir="outputs"):
    for doi in dois:
        ds_id = doi.split(".")[-1]
        dataset = pangaeapy.PanDataSet(ds_id)
        print("[INFO] Dataset title:", dataset.title)
        print("[INFO] Requesting:", doi)

        # Request dataset url
        resp = requests.get(doi)
        soup = BeautifulSoup(resp.text, "lxml")
        # Get coordinates of expedition
        lat, long = get_metadata(soup)

        # Get download link to photos page
        download_link = soup.find("div", attrs={"class": "text-block top-border"}).a[
            "href"
        ]
        print("[INFO] URL to photos page:", download_link)

        # Get to photos page (page 1)
        resp = requests.get(download_link)
        photos_page = BeautifulSoup(resp.text, "lxml")

        data = scrape_dataset(photos_page)

        # Store data
        df = pd.DataFrame(data, columns=["url"])
        df["image"] = df["url"].apply(lambda url: url.split("/")[-1])
        df["long"] = long
        df["lat"] = lat
        df["site"] = dataset.events[0].label
        df["campaign"] = dataset.events[0].campaign
        df["dataset"] = dataset.title

        # Rearranging columns
        df = df[df.columns[::-1]]
        # Save to file
        file = f"{out_dir}/[scraped]-{ds_id}.csv"
        df.to_csv(file, index=False)
        print(f"[INFO] Saved at: {file}")
        print("-" * 50)


if __name__ == "__main__":
    main()
