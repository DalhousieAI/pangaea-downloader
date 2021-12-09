#!/usr/bin/env python

"""
Pangaea search and download user interface.
"""

import os
import sys

import pandas as pd

from pangaea_downloader import __meta__
from pangaea_downloader.tools import datasets, process, scraper, search


def search_and_download(query=None, output_dir="query-outputs", verbose=1):
    """
    Search `PANGAEA`_ for a query, and download datasets for each result.

    .. _PANGAEA: https://pangaea.de/

    Parameters
    ----------
    query : str, optional
        The query to search for.
        The default behaviour is to search for the list of query strings
        specified in the file ``pangaea_downloader/query_list``.
    output_dir : str, default="query-outputs"
        The output directory where downloaded datasets will be saved.
        Any existing output datasets will be skipped instead of downloaded.
    verbose : int, default=1
        Verbosity level.
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ----------------- SEARCH PANGAEA ----------------- #
    if query is not None:
        results = search.run_search_query(query, verbose=verbose)
    else:
        results = search.run_multiple_search_queries(verbose=verbose)

    # Process each result dictionary
    n_files = 0
    n_downloads = 0
    errors = []
    for i, result in enumerate(results):
        # Extract result info
        citation, url, ds_id, size, is_parent = process.get_result_info(result)
        print(f"[{i+1:5d}/{len(results)}] Processing dataset: '{citation}'. {url}")

        # Check if file already exists in downloads
        f_name = ds_id + ".csv"
        output_path = os.path.join(output_dir, f_name)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"\t[INFO] File: '{f_name}' already exists! Skipping...")
            n_files += 1
            continue

        # ------------- ASSESS DATASET TYPE ------------- #
        try:
            if is_parent:
                df_list = datasets.fetch_children(url)
                if df_list is None:
                    print(f"\t[INFO] No child datasets! Skipping {ds_id}")
                    continue
                df_list = [df for df in df_list if df is not None]
                if len(df_list) == 0:
                    print(f"\t[INFO] All children are empty! Skipping {ds_id}")
                    continue
                df = pd.concat(df_list)
            else:
                dataset_type = process.ds_type(size)
                if dataset_type == "video":
                    print(f"\t[WARNING] Video dataset! {url} skipping...")
                    continue
                elif dataset_type == "paginated":
                    df = scraper.scrape_image_data(url)
                elif dataset_type == "tabular":
                    df = datasets.fetch_child(url)
        except BaseException as err:
            if isinstance(err, KeyboardInterrupt):
                raise
            msg = f"\t[ERROR] Could not process '{citation}', {url}:\n{err}"
            print(msg)
            errors.append(msg)
            continue

        # ----------------- SAVE TO FILE ----------------- #
        if df is None:
            continue
        try:
            saved = datasets.save_df(df, output_path, level=1)
        except BaseException as err:
            # Delete partially saved file, if present
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except BaseException:
                    pass
            raise err
        n_downloads += 1 if saved else 0

    print(f"Complete! Total files downloaded: {n_downloads}.")
    print(f"Number of files previously saved: {n_files}.")
    print(f"Total dataset files: {n_files + n_downloads}")
    print(f"Number of dataset errors (excluding access): {len(errors)}.")
    for msg in errors:
        print(msg)


def get_parser():
    """
    Build CLI parser for downloading PANGAEA datasets.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import textwrap

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Download all PANGAEA datasets discovered by a search query.",
        add_help=False,
    )

    parser.add_argument(
        "--help",
        "-h",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s {version}".format(version=__meta__.version),
        help="Show program's version number and exit.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="The query string to search for and download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="query-outputs",
        help="Directory for downloaded datasets. Default is %(default)s.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help=textwrap.dedent(
            """
            Increase the level of verbosity of the program. This can be
            specified multiple times, each will increase the amount of detail
            printed to the terminal. The default verbosity level is %(default)s.
        """
        ),
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help=textwrap.dedent(
            """
            Decrease the level of verbosity of the program. This can be
            specified multiple times, each will reduce the amount of detail
            printed to the terminal.
        """
        ),
    )
    return parser


def main():
    """
    Run command line interface for downloading images.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return search_and_download(**kwargs)


if __name__ == "__main__":
    main()
