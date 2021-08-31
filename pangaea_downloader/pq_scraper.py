#!/usr/bin/env python

"""
Pangaea search and download user interface.
"""

import os
import sys

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
    for i, result in enumerate(results):
        # Extract result info
        citation, url, ds_id, size, is_parent = process.get_result_info(result)
        print(f"[{i+1}] Processing dataset: '{citation}'. {url}")

        # Check if file already exists in downloads
        f_name = ds_id + ".csv"
        path = os.path.join(output_dir, f_name)
        if os.path.exists(path):
            print(f"\t[INFO] File: '{f_name}' already exists! Skipping...")
            n_files += 1
            continue

        # ------------- ASSESS DATASET TYPE ------------- #
        df = None
        df_list = None
        if is_parent:
            df_list = datasets.fetch_children(url)
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
        if df is not None:
            datasets.save_df(df, ds_id, output_dir)
            n_downloads += 1
        if df_list is not None:
            for child in df_list:
                child_id = child["DOI"].iloc[0].split(".")[-1]
                datasets.save_df(child, child_id, output_dir)
                n_downloads += 1

    print(f"Complete! Total files downloaded: {n_downloads}.")
    print(f"Number of files previously saved: {n_files}.")
    print(f"Total dataset files: {n_files + n_downloads}")


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
        description="Download all PANGEAEA datasets discovered by a search query.",
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
