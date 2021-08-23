#!/usr/bin/env python

"""
Pangaea search and download user interface.
"""

import os

from pangaea_downloader.tools import datasets, process, scraper, search


def main(query=None, out_dir="../query-outputs"):
    """
    Search `PANGAEA`_ for a query, and download datasets for each result.

    .. _PANGAEA: https://pangaea.de/

    Parameters
    ----------
    query : str, optional
        The query to search for.
        The default behaviour is to search for the list of query strings
        specified in the file ``pangaea_downloader/query_list``.
    out_dir : str, default="../query-outputs"
        The output directory where downloaded datasets will be saved.
        Any existing output datasets will be skipped instead of downloaded.
    """
    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # ----------------- SEARCH PANGAEA ----------------- #
    if query is not None:
        results = search.run_search_query(query, verbose=True)
    else:
        results = search.run_multiple_search_queries(verbose=True)

    # Process each result dictionary
    n_files = 0
    n_downloads = 0
    for i, result in enumerate(results):
        # Extract result info
        citation, url, ds_id, size, is_parent = process.get_result_info(result)
        print(f"[{i+1}] Processing dataset: '{citation}'. {url}")

        # Check if file already exists in downloads
        f_name = ds_id + ".csv"
        path = os.path.join(out_dir, f_name)
        if os.path.exists(path):
            print(f"\t[INFO] File: '{f_name}' already exists! Skipping...")
            n_files += 1
            continue

        # ------------- ASSESS DATASET TYPE ------------- #
        df = None
        if is_parent:
            df = datasets.fetch_children(url)
        else:
            dataset_type = process.ds_type(size)
            if dataset_type == "video":
                print(f"\t[WARNING] Video dataset! {url} skipping...")
                continue
            elif dataset_type == "paginated":
                df = scraper.scrape_image_data(url)
            elif dataset_type == "tabular":
                df = datasets.fetch_child(url)

        # ----------------- SAVE TO FILE ----------------- #
        if df is None:
            continue
        else:
            f_name = ds_id + ".csv"
            path = os.path.join(out_dir, f_name)
            df.to_csv(path, index=False)
            print(f"\t[INFO] Saved to '{path}'")
            n_downloads += 1

    print(f"Complete! Total files downloaded: {n_downloads}.")
    print(f"Number of files previously saved: {n_files}.")
    print(f"Total dataset files: {n_files + n_downloads}")


if __name__ == "__main__":
    main(out_dir="../query-outputs-new")
