import os

from pangaea_downloader.utilz import (
    fetch_child_datasets,
    fetch_dataset,
    get_result_info,
    scrape_images,
    search_pangaea,
)


def main(query="seabed photographs", n_results=10, out_dir="../query-outputs"):
    """Search www.pangaea.de for a given query string and download datasets for each result item."""
    print(f"[INFO] Running search with query: '{query}'")
    results = search_pangaea(query=query, n_results=n_results)
    os.makedirs(out_dir, exist_ok=True)
    print("[INFO] Processing results...\n")

    n_downloads = 0
    for i, result in enumerate(results):
        # Extract result information
        citation, url, size, is_parent = get_result_info(result)
        ds_id = result["URI"].split("PANGAEA.")[-1]
        print(f"[{i+1}] Loading dataset: '{citation}'")

        # ------------- ASSESS DATASET TYPE ------------- #
        df = None
        # Video dataset (ignore)
        if "bytes" in size:
            print("\t[WARNING] VIDEO dataset. Skipping...")
            continue

        # Paginated images (scrape urls and metadata)
        elif "unknown" == size:
            df = scrape_images(url)

        # Parent dataset (fetch child datasets)
        elif "datasets" in size:
            df = fetch_child_datasets(url)

        # Tabular dataset (fetch and save)
        elif "data points" in size:
            df = fetch_dataset(url)

        # ----------------- SAVE TO FILE ----------------- #
        if df is None:
            continue
        else:
            f_name = ds_id + ".csv"
            path = os.path.join(out_dir, f_name)
            df.to_csv(path, index=False)
            print(f"\t[INFO] Saved to '{path}'")
            n_downloads += 1

    print(f"COMPLETE! Total files downloaded: {n_downloads}")


if __name__ == "__main__":
    main(query="seafloor photography", n_results=999)