#!/usr/bin/env python

"""
Merge benthic PANGAEA datasets together, in BenthicNet format.

Search results are filtered to ensure they are images of the seafloor.
"""

import datetime
import os
import re
from collections import defaultdict

import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pangaea_downloader import __meta__
from pangaea_downloader.tools import checker

try:
    from benthicnet.io import fixup_repeated_output_paths
except ImportError:
    fixup_repeated_output_paths = None

TAXONOMY_RANKS = [
    ["Kingdom", "Regnum"],
    ["Phylum", "Division"],
    ["Ordo", "Order"],
    ["Familia", "Family"],
    ["Genus"],
    ["Species"],
]


def row2taxonomy(row):
    """
    Merge together taxonomical rows into a single field.
    """
    parts = []
    for rank_synonyms in TAXONOMY_RANKS:
        for col in rank_synonyms:
            col_ = col.lower()
            if col in row.keys() and row[col] and row[col] != "-":
                parts.append(row[col])
                break
            elif col_ in row.keys() and row[col_] and row[col_] != "-":
                parts.append(row[col_])
                break
        else:
            break
    return " > ".join(parts)


def find_url_column(df):
    """
    Determine the name of the column containing image URL.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    column : str or None
        The name of the best-guess column for the image URL, or None if no
        such column could be found.
    """
    # Change to lower case and strip away any non-alphanumeric characters which
    # would stop us seeing an appropriate column
    clean_cols = [
        col.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
        for col in df.columns
    ]
    # Ordered list of priorities
    # Exclude url meta/ref/source which are not links to images
    candidates = [
        "urlimage",
        "urlraw",
        "urlfile",
        "url",
        "urlgraphic",
        "urlthumb",
        "urlthumbnail",
        "image",
        "imagery",
    ]
    for candidate in candidates:
        if candidate not in clean_cols:
            continue
        col = df.columns[clean_cols.index(candidate)]
        if any(df[col].apply(checker.is_url)):
            return col


def add_file_extension(row):
    """
    Add file extension to image filename.

    Parameters
    ----------
    row : dict
        A dict record which may have fields ``"image"``, ``"File format"``,
        and ``"File type"``.

    Returns
    -------
    fname : str
        File name with extension included.
    """
    # If there is no image column, there is nothing to add an extension to
    if (
        "image" not in row.keys()
        or not row["image"]
        or not isinstance(row["image"], str)
    ):
        return ""

    # Check to see if the image column has an extension. If it seems like it
    # does, it might actually just be a dot in the middle of the file name,
    # so we check the perceived extension to make sure it's actually an
    # extension-like string
    s = row["image"]
    ext = os.path.splitext(s)[-1]
    if (
        ext.lower()
        in checker.VALID_IMG_EXTENSIONS
        + checker.INVALID_FILE_EXTENSIONS
        + checker.COMPRESSED_FILE_EXTENSIONS
    ):
        return s

    # If there was no extension found on the image, check to see if there is
    # a File format or File type field.
    for col in ["File format", "File type"]:
        if col not in row.keys():
            continue
        new_ext = row[col]
        if not new_ext or not isinstance(new_ext, str):
            continue
        new_ext = "." + new_ext.strip().lstrip(".")
        if ext == new_ext:
            break
        s += new_ext
        break

    return s


def check_title(title):
    """
    Check dataset title is acceptable for benthic habitat imagery.

    Parameters
    ----------
    title : str
        The title of the dataset.

    Returns
    -------
    bool
        Whether the dataset title is acceptable.
    """
    title = str(title)

    if "do not use" in title.lower():
        return False
    if title.startswith("Meteorological observations"):
        return False
    if title.startswith("Sea ice conditions"):
        return False
    if "topsoil" in title.lower():
        return False
    if "core" in title.lower():
        # return False
        pass
    if "aquarium" in title.lower():
        return False
    if " of the early life history " in title.lower():
        return False
    if "grab sample" in title.lower():
        return False
    if title.startswith("Calyx growth"):
        return False
    if "dried glass sponges" in title.lower():
        return False
    if "fresh glass sponges" in title.lower():
        return False
    if "spicule preparations" in title.lower():
        return False
    if title.startswith("Shell growth increments"):
        return False
    if title.startswith("Images of shell cross sections"):
        return False

    return True


def reformat_df(df, remove_duplicate_columns=True):
    """
    Reformat a PANGAEA dataset to have standardized column names.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset.

    Returns
    -------
    df : pandas.DataFrame or None
        Cleaned dataset, or ``None`` if the dataset is invalid.
    """
    # Check the title is okay, otherwise don't bother cleaning the dataframe
    if (
        "dataset_title" in df
        and len(df) > 0
        and df.iloc[0]["dataset_title"]
        and not check_title(df.iloc[0]["dataset_title"])
    ):
        return None

    # Make a copy of the dataframe so we can't overwrite the input
    df = df.copy()

    # Remove bad columns
    df.drop(labels=["-"], axis="columns", inplace=True, errors="ignore")
    # Remove duplicately named columns
    cols_to_drop = []
    if remove_duplicate_columns:
        for col in df.columns:
            if len(col) < 2:
                continue
            if (
                (col[-2] in " _")
                and (col[-1] in "123456789")
                and (col[:-2] in df.columns)
            ):
                cols_to_drop.append(col)
        df.drop(labels=cols_to_drop, axis="columns", inplace=True)

    # Find the correct URL column, and drop other columns containing "url"
    cols_to_drop = []
    mapping = {}
    col_url = find_url_column(df)
    mapping[col_url] = "url"
    for col in df.columns:
        if col != col_url and "url" in col.lower():
            cols_to_drop.append(col)

    # Search for matches to canonical columns.
    # Each entry in desired_columns is a key, value pair where the key
    # is the output column name, and the value is a list of search names
    # in order of priority. The first match will be kept and others discarded.
    desired_columns = {
        "dataset": ["ds_id", "dataset", "Campaign", "campaign"],
        "site": ["Event", "event", "Site", "site", "deployment"],
        "image": ["image", "filename"],
        "datetime": [
            "Date/Time",
            "datetime",
            "timestamp",
            "date/timestart",
            "date/timeend",
            "date",
        ],
        "latitude": [
            "Latitude",
            "latitude",
            "lat",
            "latitude+",
            "latitudemed",
            "latitudenorth",
            "latitudesouth",
        ],
        "longitude": [
            "Longitude",
            "longitude",
            "lon",
            "long",
            "longitude+",
            "longitudemed",
            "longitudewest",
            "longitudeeast",
        ],
        "x_pos": [],
        "y_pos": [],
        "altitude": ["altitude", "height"],
        "depth": [
            "depthwater",
            "bathydepth",
            "bathymetry",
            "bathy",
            "depth",
            "elevation",
        ],
        "backscatter": [],
        "temperature": ["temperature", "temp"],
        "salinity": ["salinity", "sal"],
        "chlorophyll": [],
        "acidity": ["pH"],
        "doi": ["DOI", "doi"],
    }
    # Remove non-alphanumeric padding characters, including spaces, from actual column names
    raw_cols = list(df.columns)
    clean_cols = [
        col.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
        for col in df.columns
    ]
    # Map to lower case
    lower_cols = [col.lower() for col in clean_cols]

    # Search for matching column names
    for canon, searches in desired_columns.items():
        found = False

        # Check for case-sensitive, non-alphanumeric, match
        for search in searches:
            if search not in raw_cols:
                continue
            col = search
            if not found:
                found = True
                mapping[col] = canon
                if col != canon and canon in df.columns:
                    cols_to_drop.append(canon)
            elif col not in mapping and col not in cols_to_drop:
                cols_to_drop.append(col)

        # Check for case-sensitive match
        for search in searches:
            if search not in clean_cols:
                continue
            col = df.columns[clean_cols.index(search)]
            if not found:
                found = True
                mapping[col] = canon
                if col != canon and canon in df.columns:
                    cols_to_drop.append(canon)
            elif col not in mapping and col not in cols_to_drop:
                cols_to_drop.append(col)

        # Check for case-insensitive match
        for search in searches:
            if search.lower() not in lower_cols:
                continue
            col = df.columns[lower_cols.index(search.lower())]
            if not found:
                found = True
                mapping[col] = canon
                if col != canon and canon in df.columns:
                    cols_to_drop.append(canon)
            elif col not in mapping and col not in cols_to_drop:
                cols_to_drop.append(col)

    # Remove superfluous columns
    df.drop(labels=cols_to_drop, axis="columns", inplace=True)
    # Rename columns to canonical names
    df.rename(columns=mapping, inplace=True, errors="raise")

    # Add file extension to image
    df["image"] = df.apply(add_file_extension, axis=1)
    # if "timestamp" not in df.columns and "datetime" in df.columns:
    #     df["timestamp"] = df["datetime"].apply(datetime2timestamp)

    if any([c in clean_cols for c in ["Kingdom", "Phylum", "Genus"]]):
        df["taxonomy"] = df.apply(row2taxonomy, axis=1)
        df.drop(
            labels=[x for syn in TAXONOMY_RANKS for x in syn],
            axis="columns",
            inplace=True,
            errors="ignore",
        )

    cols_to_drop = [
        "File format",
        "File type",
        "File size",
        "Date/Time",
        "Date/time end",
    ]
    df.drop(labels=cols_to_drop, axis="columns", inplace=True, errors="ignore")

    return df


def check_image_url(url):
    """
    Check image URL is acceptable for benthic habitat imagery.

    Parameters
    ----------
    url : str
        The image url.

    Returns
    -------
    bool
        Whether the image URL is acceptable.
    """
    banned_subdomains = [
        "https://doi.org/10.1594/PANGAEA",
        "http://epic.awi.de/",
        "https://epic.awi.de/",
        "http://hdl.handle.net/10013/",
        "http://library.ucsd.edu/dc/object/",
        "https://app.geosamples.org/uploads/UHM/",
        "https://hs.pangaea.de/Images/Linescan/",
        "https://hs.pangaea.de/Maps/",
        "https://hs.pangaea.de//Maps",
        "https://hs.pangaea.de/Movies/",
        "https://hs.pangaea.de/Projects/",
        "https://hs.pangaea.de/bathy/",
        "https://hs.pangaea.de/fishsounder/",
        "https://hs.pangaea.de/mag/",
        "https://hs.pangaea.de/model/",
        "https://hs.pangaea.de/nav/",
        "https://hs.pangaea.de/palaoa/",
        "https://hs.pangaea.de/pasata/",
        "https://hs.pangaea.de/para/",
        "https://hs.pangaea.de/polar",
        "https://hs.pangaea.de/reflec/",
        "https://hs.pangaea.de/sat/",
        "https://prr.osu.edu/collection/object/",
        "https://store.pangaea.de/Projects/",  # Not all bad, but mostly
        "https://store.pangaea.de/Publications/",  # Not all bad, but mostly
        "https://store.pangaea.de/software/",
        "https://www.ngdc.noaa.gov/geosamples/",
        "https://hs.pangaea.de/Images/Airphoto/",
        # "https://hs.pangaea.de/Images/Cores/",  # Some of these are okay
        "https://hs.pangaea.de/Images/Documentation/",
        "https://hs.pangaea.de/Images/Maps/",
        "https://hs.pangaea.de/Images/MMT/",
        "https://hs.pangaea.de/Images/Plankton/",
        # The GeoB19346-1 dataset contains .bmp images of the ROV's sonar
        "https://hs.pangaea.de/Images/ROV/M/M114/GeoB19346-1/data_publish/data/sonar/",
        "https://hs.pangaea.de/Images/Satellite/",
        "https://hs.pangaea.de/Images/SeaIce/",
        "https://hs.pangaea.de/Images/Water/",
        "https://store.pangaea.de/Images/Airphoto/",
        "https://store.pangaea.de/Images/Documentation/",
        "https://hs.pangaea.de/Images/Benthos/AWI_experimental_aquarium_system/",
        # "https://hs.pangaea.de/Images/Benthos/AntGlassSponges",  # Only okay if it contains "AHEAD"
        "https://hs.pangaea.de/Images/Benthos/Kongsfjorden/MHerrmann/",  # Microscope images
        "https://hs.pangaea.de/Images/Benthos/Kongsfjorden/Brandal/",  # Cross-sections
    ]
    banned_words = [
        "aquarium",
        "map",
        "divemap",
        "dredgephotos",
        "dredge_photos",
        "dredgephotograph",
        "grabsample",
        "grab_sample",
    ]

    # Check if the URL is on any of the banned subdomains known to contain the
    # wrong sort of images
    for entry in banned_subdomains:
        if url.startswith(entry):
            return False

    # Check if the URL contains a banned word indicating it is of metadata or
    # samples taken away from the seafloor
    for word in banned_words:
        if re.search("(?<![A-Za-z])" + word + "(?![A-Za-z])", url.lower()):
            return False

    if re.search("(?<![a-z])core(?![a-rty])", url.lower()) and "SUR" not in url:
        # Images of cores must contain "SURFACE", or the shorthand "SUR"
        # We only keep the ones with surface in uppercase, because those
        # experiments are in-situ surface photos, whereas lower case are not.
        return False

    if (
        url.startswith("https://hs.pangaea.de/Images/Benthos/AntGlassSponges/")
        and "AHEAD" not in url
    ):
        # Images of AntGlassSponges must contain "AHEAD" to be kept
        # otherwise, they are of sponges after removal
        return False

    if "not_available" in url:
        return False

    return True


def filter_urls(df, url_column="url", inplace=True):
    """
    Remove unwanted URLs from the dataset.

    Remove invalid URLs (file paths, etc), filter out off-topic URLs by their
    subdomain and checking for blacklisted keywords, and remove URLs which
    are not images. We also remove any mosaic images.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to filter.
    url_column : str, default="url"
        The name of the column containing image URLs, which will be filtered.
    inplace : bool, default=True
        Whether to adjust the input dataframe, otherwise a copy will be used.

    Returns
    -------
    df : pandas.DataFrame
        The dataset, with rows removed if they were deemed to have bad URLs.
    """
    if not inplace:
        df = df.copy()

    if df is None or len(df) == 0:
        return df

    # Filter down to only valid URLs
    df = df[df[url_column].apply(checker.is_url)]
    if df is None or len(df) == 0:
        return df

    # Remove bad subdomains and blacklisted words
    df = df[df[url_column].apply(check_image_url)]
    if df is None or len(df) == 0:
        return df

    # Filter down to only rows which have image extension
    is_image = df[url_column].apply(lambda x: checker.has_img_extension(x.rstrip("/")))
    if "image" in df.columns:
        is_image |= df["image"].apply(
            lambda x: checker.has_img_extension(x.rstrip("/"))
        )
    df = df[is_image]
    if df is None or len(df) == 0:
        return df

    # Drop mosaic images
    df = df[~df[url_column].apply(lambda x: "mosaic" in x.lower())]

    return df


def fixup_url_with_image(row):
    """
    Replace the end of the URL field with the contents of the image field.

    Parameters
    ----------
    row : dict
        Dictionary containing one record, with keys ``"url"`` and (optionally)
        ``"image"``. If there is no ``"image"`` key, the URL will not be
        changed.

    Returns
    -------
    url_new : str
        New image URL.
    """
    url_old = row["url"]
    if "image" not in row or not row["image"]:
        return url_old
    url = url_old.rstrip("/ ")
    url_parts = url.split("/")
    ext = os.path.splitext(url_parts[-1])[-1]
    url_new = "/".join(url_parts[:-1])
    if row["image"] + ext == url_parts[-1]:
        return url
    url_new += "/" + row["image"]
    return url_new


def insert_rows(df, rows, indices):
    """
    Insert rows into a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to add rows to.
    rows : list of dict
        The rows to add to the DataFrame.
    indices : list of int
        The indices at which the rows will be inserted.

    Returns
    -------
    df : pandas.DataFrame
        New dataframe, with rows added.
    """
    parts = [df[: indices[0]], pd.DataFrame([rows[0]])]
    for j in range(len(indices) - 1):
        parts.append(df[indices[j - 1] : indices[j]])
        parts.append(pd.DataFrame([rows[j]]))
    parts.append(df[indices[-1] :])
    return pd.concat(parts)


def fixup_repeated_urls(
    df, url_column="url", inplace=True, force_keep_original=True, verbose=1
):
    """
    Change rows with repeated URLs to use their image field for the URL instead.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to process.
    url_column : str, default="url"
        Name of the column in ``df`` containing image URLs.
    inplace : bool, default=True
        Whether to update the input ``df``. Otherwise a copy is returned.
    force_keep_original : bool, default=True
        Whether to ensure that original URLs still exist in the output by
        copying the first row containing a repeated URL if all original
        duplicates found new URLs.
    verbose : int, default=1
        Level of verbosity.

    Returns
    -------
    df : pandas.DataFrame
        The dataset, with repeated URLs adjusted to use their image field
        instead.
    """
    if not inplace:
        df = df.copy()
    dup_urls = df[df.duplicated(subset=url_column)][url_column].unique()
    if "image" not in df.columns:
        if len(dup_urls) and verbose >= 1:
            print(
                f"Can't clean {len(dup_urls)} repeated URLs in {df.loc[0, 'ds_id']} without 'image' column"
            )
        return df
    if len(dup_urls) == 0:
        return df
    if verbose >= 1 and "dataset" in df.columns:
        print(f"{df.iloc[0]['dataset']} has {len(dup_urls)} duplicated URLs")
    rows_to_insert = []
    indices_to_insert_at = []
    for dup_url in dup_urls:
        if pd.isna(dup_url):
            continue
        is_bad_url = df[url_column] == dup_url
        if verbose >= 2:
            n_to_change = sum(is_bad_url)
            print(f"Fixing up {n_to_change} repetitions of the URL {dup_url}")
        first_row_idx = np.nonzero(is_bad_url.values)[0][0]
        if force_keep_original:
            # first_row = df[is_bad_url].iloc[0].copy()
            first_row = df.iloc[first_row_idx].copy()
        df.loc[is_bad_url, url_column] = df[is_bad_url].apply(
            fixup_url_with_image, axis=1
        )
        n_remain = sum(df.loc[is_bad_url, url_column] == dup_url)
        if verbose >= 2:
            if n_remain == n_to_change:
                print(f"  All {n_to_change} URLs left unchanged")
            else:
                print(f"  {n_remain} / {n_to_change} URLs remain unchanged")
                if verbose >= 3:
                    print("  After:")
                    for x in df.loc[is_bad_url, url_column][:5]:
                        print(f"    {x}")
                    if n_to_change > 5:
                        print("    ...")
                        print("    " + df.loc[is_bad_url, url_column].values[-1])

        if force_keep_original and n_remain == 0:
            if verbose >= 1:
                print(f"  Duplicating a row since the URL {dup_url} no longer appears")
            first_row[url_column] = dup_url
            rows_to_insert.append(first_row)
            indices_to_insert_at.append(first_row_idx)
    if len(rows_to_insert) > 0:
        if verbose >= 1:
            print(f"  Inserting {len(rows_to_insert)} duplicated rows")
        df = insert_rows(df, rows_to_insert, indices_to_insert_at)
    return df


def process_datasets(input_dirname, output_path=None, verbose=0):
    """
    Process a directory of datasets: clean, concatenate and save.

    Parameters
    ----------
    input_dirname : str
        Path to directory containing CSV files.
    output_path : str, optional
        The output filename. Default is ``input_dirname + ".csv"``.
    verbose : int, default=0
        Level of verbosity.
    """
    if output_path is None:
        output_path = input_dirname.rstrip(r"\/") + ".csv"

    if verbose >= 0:
        print(f"Processing directory {input_dirname}")
        print(f"Will save as {output_path}")

    column_count = defaultdict(lambda: 0)
    column_examples = defaultdict(lambda: [])
    files_without_url = []
    files_with_repeat_urls = []
    files_with_repeat_urls2 = []
    n_total = 0
    n_valid = 0
    dfs = []
    dfs_fnames = []

    for fname in tqdm(sorted(sorted(os.listdir(input_dirname)), key=len)):  # noqa: C414
        if not fname.endswith(".csv"):
            continue
        # for fname in tqdm(os.listdir(input_dirname)):
        ds_id = os.path.splitext(fname)[0]
        df = pd.read_csv(os.path.join(input_dirname, fname))
        n_total += 1
        if not checker.has_url_col(df):
            continue

        url_col = find_url_column(df)
        if not url_col:
            files_without_url.append(fname)
            continue

        df["ds_id"] = f"pangaea-{ds_id}"
        df = reformat_df(df)
        if df is None:
            continue

        url_col = "url"
        df = df[df[url_col] != ""]
        if len(df) == 0:
            continue

        df = filter_urls(df, url_column=url_col)
        if len(df) == 0:
            continue

        n_valid += 1

        for col in df.columns:
            column_count[col] += 1
            column_examples[col].append(fname)

        # Drop rows that are complete duplicates
        df.drop_duplicates(inplace=True)

        if len(df) != len(df.drop_duplicates(subset=url_col)):
            files_with_repeat_urls.append(fname)

        # Try to fix repeated URLs that are accidental dups but should differ
        df = fixup_repeated_urls(df, url_column=url_col, verbose=1)

        if len(df) != len(df.drop_duplicates(subset=url_col)):
            files_with_repeat_urls2.append(fname)

        # Check for any rows that are all NaNs
        if sum(df.isna().all("columns")) > 0:
            print(f"{ds_id} has a row which is all NaNs")

        dfs.append(df)
        dfs_fnames.append(fname)

    print(f"There are {n_valid} valid (of {n_total}) valid datasets")
    print(
        f"Of which {len(files_with_repeat_urls)} have repeated URLs (before replacing dups with image)"
    )
    print(
        f"Of which {len(files_with_repeat_urls2)} have repeated URLs (after replacing dups with image)"
    )
    print()
    print(f"There are {len(column_count)} unique column names:")
    print()

    for col, count in dict(
        sorted(column_count.items(), key=lambda item: item[1], reverse=True)
    ).items():
        c = col + " "
        print(f"{c:.<35s} {count:4d}")
    print()

    if verbose >= 1:
        print("Filter columns")
    # Select only columns of interest
    select_cols = {
        "dataset",
        "site",
        "url",
        "image",
        "datetime",
        "latitude",
        "longitude",
        "altitude",
        "depth",
        "backscatter",
        "temperature",
        "salinity",
        "chlorophyll",
        "acidity",
    }
    df_all = pd.concat(
        [df[df.columns.intersection(select_cols)] for df in dfs if len(df) > 0]
    )

    # Remove duplicate URLs
    if verbose >= 1:
        print("Remove duplicates")
    df_all.drop_duplicates(subset="url", inplace=True, keep="first")

    # Fix repeated output paths by replacing with image field
    if fixup_repeated_output_paths is None:
        if verbose >= 1:
            print("Skip fix repeated output paths step (requires benthicnet package)")
    else:
        if verbose >= 1:
            print("Fix repeated output paths to prevent collisions")
        df_all = fixup_repeated_output_paths(df_all, inplace=True, verbose=2)

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if verbose >= 0:
        print(f"Saving to {output_path}")
    df_all.to_csv(output_path, index=False)


def get_parser():
    """
    Build CLI parser for processing PANGAEA datasets.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import sys
    import textwrap

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Process downloaded PANGAEA datasets.",
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
        "input_dirname",
        type=str,
        help="The query string(s) to search for and download.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output CSV file name.",
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
    Run command line interface for cleaning and merging datasets.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return process_datasets(**kwargs)


if __name__ == "__main__":
    main()
