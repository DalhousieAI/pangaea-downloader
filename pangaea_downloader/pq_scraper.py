#!/usr/bin/env python

"""
Pangaea search and download user interface.
"""

import os
import sys
import traceback

import colorama
import pandas as pd
from tqdm.auto import tqdm

from pangaea_downloader import __meta__
from pangaea_downloader.tools import datasets, process, scraper, search


def search_and_download(
    queries=None,
    output_dir="query-outputs",
    auth_token=None,
    verbose=0,
):
    """
    Search `PANGAEA`_ for a set of queries, and download datasets for each result.

    .. _PANGAEA: https://pangaea.de/

    Parameters
    ----------
    queries : str or list of str, optional
        The queries to search for.
        The default behaviour is to search for the list of query strings
        specified in the file ``pangaea_downloader/query_list``.
    output_dir : str, default="query-outputs"
        The output directory where downloaded datasets will be saved.
        Any existing output datasets will be skipped instead of downloaded.
    auth_token : str, optional
        Bearer authentication token.
    verbose : int, default=1
        Verbosity level.
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ----------------- SEARCH PANGAEA ----------------- #
    if queries is None or len(queries) == 0:
        # Read in file containing default list of search queries
        queries = search.read_query_list()
    elif isinstance(queries, str):
        queries = [queries]
    results = search.run_multiple_search_queries(queries, verbose=verbose)

    df_results = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(output_dir.rstrip("/") + "_search_results.csv", index=False)

    # Process each result dictionary
    n_files = 0
    n_downloads = 0
    errors = []
    for i, result in enumerate(tqdm(results, disable=verbose != 0)):
        # Extract result info
        citation, url, ds_id, size, is_parent = process.get_result_info(result)
        if verbose >= 1:
            print(f"[{i+1:5d}/{len(results)}] Processing dataset: '{citation}'. {url}")

        # Check if file already exists in downloads
        f_name = ds_id + ".csv"
        output_path = os.path.join(output_dir, f_name)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            if verbose >= 1:
                print(
                    colorama.Fore.CYAN
                    + f"\t[INFO] File: '{f_name}' already exists! Skipping..."
                    + colorama.Fore.RESET
                )
            n_files += 1
            continue

        # ------------- ASSESS DATASET TYPE ------------- #
        try:
            if is_parent:
                df_list = datasets.fetch_children(
                    url,
                    verbose=verbose - 1,
                    auth_token=auth_token,
                )
                if df_list is None:
                    if verbose >= 1:
                        print(
                            colorama.Fore.CYAN
                            + f"\t[INFO] No child datasets! Skipping {ds_id}"
                            + colorama.Fore.RESET
                        )
                    continue
                df_list = [df for df in df_list if df is not None]
                if len(df_list) == 0:
                    if verbose >= 1:
                        print(
                            colorama.Fore.CYAN
                            + f"\t[INFO] All children are empty! Skipping {ds_id}"
                            + colorama.Fore.RESET
                        )
                    continue
                df = pd.concat(df_list)
            else:
                try:
                    dataset_type = process.ds_type(size)
                except Exception:
                    raise ValueError(f"Can't process type from size for {ds_id}")
                if dataset_type == "video":
                    if verbose >= 1:
                        print(
                            colorama.Fore.YELLOW
                            + f"\t[WARNING] Video dataset! {url} skipping..."
                            + colorama.Fore.RESET
                        )
                    continue
                elif dataset_type == "paginated":
                    df = scraper.scrape_image_data(url, verbose=verbose - 1)
                elif dataset_type == "tabular":
                    df = datasets.fetch_child(
                        url,
                        verbose=verbose - 1,
                        auth_token=auth_token,
                    )
        except Exception as err:
            if isinstance(err, KeyboardInterrupt):
                raise
            msg = f"\t[ERROR] Could not process '{citation}', {url}\n{err}"
            msg += "\n\n" + traceback.format_exc()
            errors.append(traceback.format_exc())
            if verbose >= 0:
                print(colorama.Fore.RED + msg + colorama.Fore.RESET)
            continue

        # ----------------- SAVE TO FILE ----------------- #
        if df is None:
            continue
        try:
            saved = datasets.save_df(df, output_path, level=1, verbose=verbose - 1)
        except Exception as err:
            # Delete partially saved file, if present
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            raise err
        n_downloads += 1 if saved else 0

    if verbose >= 0:
        print(f"Complete! Total files downloaded: {n_downloads}.")
        print(f"Number of files previously saved: {n_files}.")
        print(f"Total dataset files: {n_files + n_downloads}")
        print(f"Number of dataset errors (excluding access): {len(errors)}.")
        if len(errors) > 0:
            print()
            print("Captured errors are now repeated as follows.")
        for msg in errors:
            print()
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
        nargs="+",
        help="The query string(s) to search for and download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="query-outputs",
        help="Directory for downloaded datasets. Default is %(default)s.",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        help="Bearer authentication token",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
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
    kwargs["queries"] = kwargs.pop("query")

    return search_and_download(**kwargs)


if __name__ == "__main__":
    main()
