#!/usr/bin/env python

"""
Merge benthic PANGAEA datasets together, in BenthicNet format.

Search results are filtered to ensure they are images of the seafloor.
"""

import datetime
import os
import re
from collections import defaultdict
from functools import partial

import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
from pangaeapy import PanDataSet
from tqdm.auto import tqdm

from pangaea_downloader import __meta__
from pangaea_downloader.tools import checker

try:
    from benthicnet.io import fixup_repeated_output_paths, row2basename
except ImportError:
    fixup_repeated_output_paths = None
    row2basename = None

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
    if (
        "early biofouling processes in a coastal lagoon" in title.lower()
        or "early biofouling processes in a coastal la goon" in title.lower()
    ):
        return False
    if "photographs of tiles" in title.lower():
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
        "dataset": ["ds_id"],
        "site": ["Event", "event", "deployment"],
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
            # "latitudesouth",  # special handling
        ],
        "longitude": [
            "Longitude",
            "longitude",
            "lon",
            "long",
            "longitude+",
            "longitudemed",
            "longitudeeast",
            # "longitudewest",  # special handling
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

    # Handle latitudesouth and longitudewest
    if "latitude" not in df.columns and "latitudesouth" in df.columns:
        df["latitude"] = -df["latitudesouth"]
    if "latitude" not in df.columns and "latitude-" in df.columns:
        df["latitude"] = -df["latitude-"]
    if "longitude" not in df.columns and "longitudewest" in df.columns:
        df["longitude"] = -df["longitudewest"]
    if "longitude" not in df.columns and "longitude-" in df.columns:
        df["longitude"] = -df["longitude-"]

    # Add file extension to image
    df["image"] = df.apply(add_file_extension, axis=1)
    # if "timestamp" not in df.columns and "datetime" in df.columns:
    #     df["timestamp"] = df["datetime"].apply(datetime2timestamp)

    # Add default site if it is missing
    if "site" not in df.columns:
        df["site"] = df["dataset"] + "_site"

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


def fixup_favourite_images(df, verbose=1):
    """
    Drop duplicated favourite images.

    These occur in Ingo Schewe's datasets along OFOS profiles during POLARSTERN
    cruises, PANGAEA dataset ids 849814--849816 and 873995--874002.

    Parameters
    ----------
    df : pandas.DataFrame
        A PANGAEA dataframe with Type column.
    verbose : int, default=1
        Level of verbosity.

    Returns
    -------
    df : pandas.DataFrame
        As input dataframe, but with all Type entries starting with favourite
        removed (case-insensitive).
    """
    if "Type" not in df.columns:
        return df
    # Remove all Favourite timer, Favourite hotkey, FAVOURITE_TIMER, and
    # FAVOURITE_HOTKEY entries, which although they have unique URLs for their
    # images are actually identical images to the ones occuring immediately
    # after them in the dataframe.
    n_samples_before = len(df)
    df = df[~df["Type"].str.lower().str.startswith("favourite")]
    n_samples_after = len(df)
    if verbose >= 1 and n_samples_after != n_samples_before:
        print(
            f"{df.iloc[0]['dataset']}:"
            f" Removed {n_samples_before - n_samples_after} favourited images."
            f" {n_samples_before} -> {n_samples_after} rows"
        )
    return df


def get_dataset_datetime(ds_id):
    """
    Determine a generic date for a dataset from the min and max extent datetimes.

    Parameters
    ----------
    ds_id : int
        The identifier of a PANGAEA dataset.

    Returns
    -------
    dt_avg : str
        The average datetime between the min and max extent, with precision
        reduced to reflect what can accurately be represented.
    """
    ds = PanDataSet(ds_id)
    dt_min = pd.to_datetime(ds.mintimeextent)
    dt_max = pd.to_datetime(ds.maxtimeextent)
    if dt_min is None and dt_max is None:
        return pd.NaT
    elif dt_min is None:
        return dt_max.strftime("%Y-%m-%d")
    elif dt_max is None:
        return dt_min.strftime("%Y-%m-%d")
    delta = dt_max - dt_min
    dt_avg = dt_min + delta / 2
    if delta > datetime.timedelta(days=90):
        return dt_avg.strftime("%Y")
    if delta > datetime.timedelta(days=4):
        return dt_avg.strftime("%Y-%m")
    if delta > datetime.timedelta(hours=3):
        return dt_avg.strftime("%Y-%m-%d")
    if delta > datetime.timedelta(minutes=5):
        return dt_avg.strftime("%Y-%m-%d %H:00:00")
    if delta > datetime.timedelta(seconds=5):
        return dt_avg.strftime("%Y-%m-%d %H:%M:00")
    return dt_avg.strftime("%Y-%m-%d %H:%M:%S")


def fix_missing_datetime_from_image_name(df, ds_id, verbose=1):
    """
    Extract datetime information from the contents of the image column in the dataframe.

    Note that the extraction operation is only performed on dataset IDs for
    which the image naming scheme has been manually evaluated, and is not
    applied blindly to datasets which have not been inspected.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    ds_id : int
        The identifier of the PANGAEA dataset.
    verbose : int, default=1
        Verbosity level.

    Returns
    -------
    df : pandas.DataFrame
        As input, but with missing datetime cells filled in from the image.
        Existing datetime values are unchanged.
    """
    if "datetime" not in df.columns:
        df["datetime"] = pd.NaT

    ds_id = int(ds_id)

    select = df["datetime"].isna()

    if row2basename is None:
        selected_image = df.loc[select, "image"]
    else:
        selected_image = df[select].apply(
            partial(row2basename, use_url_extension=True), axis=1
        )

    selected_image_no_ext = selected_image.apply(lambda x: os.path.splitext(x)[0])

    if ds_id in [
        785104,
        785105,
        785108,
        785109,
        785110,
        836457,
        867771,
        867772,
        867773,
        867774,
        867775,
        867776,
        867777,
        867778,
        867806,
        867807,
        867808,
        867852,
        867853,
        867861,
        873541,
        875713,
        875714,
        876422,
        876423,
        876511,
        876512,
        876513,
        876514,
        876515,
        876516,
        876517,
        876518,
        880043,
        880044,
        885666,
        885667,
        885668,
        885669,
        885670,
        885672,
        885674,
        885675,
        885709,
        885712,
        885713,
        885714,
        885715,
        885716,
        885717,
        885718,
        885719,
        885720,
    ]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. PP_107-100_2012-03-19.png
        # e.g. PP_100_2012-06-05a.jpg
        # e.g. TH_122_2012-03-27.jpg
        # e.g. J_05_2017_05_24a.jpg
        # e.g. J_overview_2017-05-24za.jpg
        # e.g. J_40_2017_08_11a.jpg
        # e.g. J_05_2017-08-11a.jpg
        # e.g. LG_OVERVIEW_01_05_06_07_09_2013_02_24a.jpg
        # e.g. LG_01_07_2010_11_11a.jpg
        # e.g. LG_01_2010_11_11a.jpg
        # e.g. LG_Cluster1_2012_01_31a.jpg
        # e.g. LG_01_07_2012_04_22a.jpg
        # e.g. LG_SCREW_2012_04_22a.jpg
        # e.g. So_01_2014_02_15b.jpg
        # e.g. XH_01_2013_01_12_a.jpg
        # e.g. XH_01%2B09_2013_11_19_a.jpg
        # e.g. XH_01_2010_04_22_a.jpg
        # e.g. LH_020_2015_01_28a_counted.jpg
        # e.g. LH_020_2015_01_28xx.jpg
        # e.g. J_J40%2BJ46%2BJ41_2016_09_25_a.jpg
        dtstr = selected_image_no_ext.str.lower().str.rstrip(
            "abcdefghijklmnopqrstuvwxyz_-"
        )
        dtstr = dtstr.str[-11:].str.replace("_", "-").str.lstrip("-")
        df.loc[select, "datetime"] = pd.to_datetime(dtstr, format="%Y-%m-%d")

    elif ds_id in [
        789211,
        789212,
        789213,
        789214,
        789215,
        789216,
        789219,
        819234,
    ]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. 2003_W01-2.jpg
        # e.g. 2004_B_bewachsen.jpg
        # e.g. 2005_B.jpg
        # e.g. 2013_B01-1.jpg
        dtstr = selected_image_no_ext.str[:4]
        # Test the format is correct; we will get an error if not
        _ = pd.to_datetime(dtstr, format="%Y")
        # But we actually want to keep the lower precision string
        df.loc[select, "datetime"] = dtstr

    elif ds_id in [
        789217,
        793210,
        793211,
        818906,
        818907,
        836263,
        836264,
        836265,
        836266,
        837653,
    ]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. 04_2011.jpg
        # e.g. 04a_2011_analog.jpg
        # e.g. 04.2-2008.jpg
        # e.g. 08-2008.jpg
        # e.g. 04a_2013.jpg
        # e.g. 05a_2003.jpg
        # e.g. 04_2007.jpg
        dtstr = selected_image_no_ext.str.lower().str.rstrip(
            "abcdefghijklmnopqrstuvwxyz_-"
        )
        dtstr = dtstr.str[-4:]
        # Test the format is correct; we will get an error if not
        _ = pd.to_datetime(dtstr, format="%Y")
        # But we actually want to keep the lower precision string
        df.loc[select, "datetime"] = dtstr

    elif ds_id in [836024, 836025]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. 00setting_2014-08.jpg
        # e.g. 39.9_2014.jpg
        # e.g. 2014_B01-1.jpg
        df.loc[select, "datetime"] = "2014"

    elif ds_id in [840699, 840700, 840702, 840703, 840742, 840743]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. J_001_2012-01-31.jpg
        # e.g. J_003_2012-01-31_2.jpg
        # e.g. J_115_2012-01-31_a.jpg
        # e.g. J_033_2012-08-08.jpg
        dtstr = selected_image_no_ext.apply(lambda x: x.split("_")[2])
        dtstr = dtstr.str[:10]
        df.loc[select, "datetime"] = pd.to_datetime(dtstr, format="%Y-%m-%d")

    elif ds_id in [840701, 849298]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. J_002_2013-03_03a.jpg
        # e.g. J_001_2015-01.jpg
        # e.g. J_001_2015-01_a.jpg
        # e.g. J_056_2013-03_06logger.jpg
        dtstr = selected_image_no_ext.apply(lambda x: x.split("_")[2])
        # Test the format is correct; we will get an error if not
        _ = pd.to_datetime(dtstr, format="%Y-%m")
        # But we actually want to keep the lower precision string
        df.loc[select, "datetime"] = dtstr

    elif ds_id in [872407, 872408, 872409, 872410, 872411]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. J_40_2017-01-12_a.jpg
        # e.g. J_overview2_2017-02-02_x.jpg
        # e.g. J_xx_2017-01-12_x-62.jpg
        # e.g. J_17_2017-01-14.jpg
        # e.g. J_23_2017-01-14_b-1.jpg
        dtstr = selected_image_no_ext.apply(lambda x: x.split("_")[2])
        df.loc[select, "datetime"] = pd.to_datetime(dtstr, format="%Y-%m-%d")

    elif ds_id in [878045, 888410]:
        # Nothing to do
        pass

    elif ds_id in [894734]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. HOTKEY_2018_03_27at21_09_21CP4A4682
        # e.g. TIMER_2018_03_18at04_04_09CP4A3970
        dtstr = selected_image_no_ext.apply(lambda x: "_".join(x.split("_")[1:])[:20])
        df.loc[select, "datetime"] = pd.to_datetime(dtstr, format="%Y_%m_%dat%H_%M_%S")

    elif ds_id in [896157]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. 2016-08-2600000.jpg
        dtstr = selected_image_no_ext.str[:10]
        df.loc[select, "datetime"] = pd.to_datetime(dtstr, format="%Y-%m-%d")

    if ds_id in [
        918232,
        918233,
        918327,
        918340,
        918341,
        918382,
        918383,
        918385,
    ]:
        if verbose >= 1:
            print(
                f"{ds_id}: Extracting missing datetime from filename for dataset {ds_id}"
            )
        # e.g. XH_01_2010_04_22_a.jpg
        # e.g. XH_01_2010_04_28a.jpg
        # e.g. XH_03_2018_10_18_a-1.jpg
        dtstr = selected_image_no_ext.apply(lambda x: "-".join(x.split("_")[2:5])[:10])
        df.loc[select, "datetime"] = pd.to_datetime(dtstr, format="%Y-%m-%d")

    return df


def add_missing_datetime(df, ds_id=None, verbose=1):
    """
    Add missing datetime values using either the mean extent or extraction from the file name.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    ds_id : int, optional
        The identifier of the PANGAEA dataset. The default behaviour is to
        extract this from the dataset column of the dataframe.
    verbose : int, default=1
        Verbosity level.

    Returns
    -------
    df : pandas.DataFrame
        As input, but with missing datetime cells completed, either by using the
        average from the datetime extent metadata, or by extracting it from the
        image name.
        All existing datetime values are left unchanged.
    """
    if "datetime" not in df.columns:
        df["datetime"] = pd.NaT

    if ds_id is None:
        # Get dataset id from first row
        ds_id = df.iloc[0]["dataset"].split("-")[-1]
    ds_id = int(ds_id)

    # Add datetimes that are still missing by inferring from the image filename
    df = fix_missing_datetime_from_image_name(df, ds_id, verbose=verbose)

    if all(df["datetime"].isna()):
        # This dataset has no datetime values
        # Try to determine average datetime from the datetime extent metadata on
        # the dataset record
        dt_avg = get_dataset_datetime(ds_id)
        if dt_avg is not None:
            if verbose >= 1:
                print(
                    f"{ds_id}: Using average datetime from extent"
                    f" - filenames look like {df.iloc[0]['image']}"
                )
            df["datetime"] = dt_avg

    if not any(df["datetime"].isna()):
        # This dataframe already has all datetime information
        return df

    select = df["datetime"].isna()
    if ds_id in [889035, 889025]:
        if verbose >= 1:
            print(f"{ds_id}: Adding manual missing datetime for {ds_id}")
        # From the abstract on PANGAEA (sic):
        # Experimet was setup during 2007-02-15 and 2007-06-13.
        df.loc[select, "datetime"] = "2007"

    if ds_id in [896160, 896164]:
        if verbose >= 1:
            print(f"{ds_id}: Adding manual missing datetime for {ds_id}")
        # From the INDEX 2016 ROV (see dataset title and paper
        # https://doi.org/10.3389/fmars.2019.00096)
        df.loc[select, "datetime"] = "2016"

    return df


def interpolate_by_datetime(df, columns):
    """
    Use datetime column to interpolate values for selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with ``"datetime"`` column, which may contain missing values
        in other columns.
    columns : str or iterable of str
        Name of column or columns to fill in missing values with interpolation.

    Returns
    -------
    df : pandas.DataFrame
        Like input, but with missing values in specified columns completed by
        linear interpolation over datetime.
    """
    # Convert datetime string to a datetime object
    datetime_actual = pd.to_datetime(df["datetime"])
    has_datetime = ~datetime_actual.isna()
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        has_col = ~df[col].isna()
        has_dt_and_col = has_datetime & has_col
        has_dt_not_col = has_datetime & ~has_col
        df.loc[has_dt_not_col, col] = np.interp(
            datetime_actual[has_dt_not_col],
            datetime_actual[has_dt_and_col],
            df.loc[has_dt_and_col, col],
        )
    return df


def fixup_incomplete_metadata(df, ds_id=None, verbose=1):
    """
    Fix datasets which have partial, but incomplete, lat/lon/datetime metadata.

    Interpolation is performed as appropriate to the dataset. The methodology
    was determined by manually inspecting each dataset.
    Any latitude and longitude values which can not be resolved are filled in
    with the dataset-level mean latitude and longitude as reported by PANGAEA.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    ds_id : int, optional
        The identifier of the PANGAEA dataset. The default behaviour is to
        extract this from the dataset column of the dataframe.
    verbose : int, default=1
        Verbosity level.

    Returns
    -------
    df : pandas.DataFrame
        As input, but with missing datetime, latitude, longitude, and/or depth
        cells completed by interpolation or similar.
        All existing datetime values are left unchanged.
    """
    if ds_id is None:
        # Get dataset id from first row
        ds_id = df.iloc[0]["dataset"].split("-")[-1]
    ds_id = int(ds_id)

    if ds_id in [753197]:
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
            print("Nothing to be done.")

    if ds_id in [805606, 805607, 805611, 805612]:
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
            print(f"{ds_id}: Interpolating by index")
        indices = np.arange(len(df))
        col = "datetime"
        select_not_col = df[col].isna()
        select_has_col = ~select_not_col
        if any(select_has_col) and any(select_not_col):
            missing_timestamps = np.interp(
                indices[select_not_col],
                indices[select_has_col],
                pd.to_datetime(df.loc[select_has_col, "datetime"]).apply(
                    lambda x: x.timestamp()
                ),
            )
            df.loc[select_not_col, col] = [
                datetime.datetime.fromtimestamp(int(ts)) for ts in missing_timestamps
            ]

    if ds_id == 875080:
        # N.B. There is date metadata in the csv, but not time. But there is time
        # metadata in the filename, so we could extract this if we wanted to.
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
            print("Nothing to be done.")
        # lat/lon was only recorded for the first 11 images. Fill in the rest
        # with the median latitude and longitude for the record at the end
        # of this function.

    if 873995 <= ds_id <= 874002:
        if verbose >= 1:
            print(f"Interpolating latitude, longitude, and depth for dataset {ds_id}")
        # Interpolate lat, lon, and depth based on datetime
        df = interpolate_by_datetime(df, ["latitude", "longitude", "depth"])

    if ds_id in [875071, 875073]:
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
        # Drop rows without datetime values (these have missing lat/lon as well)
        # For 875071, these images are of the deck of the ship.
        # For 875073, these images have a translation of less than half an image
        # from the subsequent image, so we don't need the ones without metadata.
        df = df[~df["datetime"].isna()]
        # Interpolate missing depth values
        df = interpolate_by_datetime(df, ["depth"])

    if ds_id in [875084]:
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
        # For 875084, images without latitude and longitude are not useful.
        # The first three are of the deck, the rest are dark watercolumn shots.
        df = df[~df["longitude"].isna()]
        # Interpolate missing depth values
        df = interpolate_by_datetime(df, ["depth"])

    if (878001 <= ds_id <= 878019) or ds_id == 878045:
        if verbose >= 1:
            print(f"{ds_id}: Dropping rows missing metadata for dataset {ds_id}")
        # Images without metadata are of the water column and highly redundant.
        df = df[~df["longitude"].isna()]

    if ds_id in [894732, 894734]:
        if verbose >= 1:
            print(f"{ds_id}: Dropping rows missing metadata for dataset {ds_id}")
        # It's not clear to me that any of these images are of the seafloor.
        df = df[~df["longitude"].isna()]

    if ds_id in [895557, 903782, 903788, 903850, 907025, 894801]:
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
            print(
                f"{ds_id}: Interpolating by index over subset of images in the same series"
            )
        indices = np.arange(len(df))
        image_no_ext = df["image"].apply(lambda x: os.path.splitext(x)[0])
        image_major = image_no_ext.str[:-3]
        missing_dt = df["datetime"].isna()
        missing_lat = df["latitude"].isna()
        missing_lon = df["longitude"].isna()
        for image_major_i in image_major.unique():
            select = image_major == image_major_i
            col = "latitude"
            select_and_col = select & ~missing_lat
            select_not_col = select & missing_lat
            if any(select_and_col) and any(select_not_col):
                df.loc[select_not_col, col] = np.interp(
                    indices[select_not_col],
                    indices[select_and_col],
                    df.loc[select_and_col, col],
                )
            col = "longitude"
            select_and_col = select & ~missing_lon
            select_not_col = select & missing_lon
            if any(select_and_col) and any(select_not_col):
                df.loc[select_not_col, col] = np.interp(
                    indices[select_not_col],
                    indices[select_and_col],
                    df.loc[select_and_col, col],
                )
            col = "datetime"
            select_and_col = select & ~missing_dt
            select_not_col = select & missing_dt
            if any(select_and_col) and any(select_not_col):
                df.loc[select_not_col, col] = scipy.interpolate.interp1d(
                    indices[select_and_col],
                    pd.to_datetime(df.loc[select_and_col, col]),
                    kind="nearest",
                    fill_value="extrapolate",
                )(indices[select_not_col])

    if ds_id in [911904, 918924, 919348]:
        if verbose >= 1:
            print(f"{ds_id}: Extracting missing datetime metadata for dataset {ds_id}")
        # Extract missing datetime from the filename, formatted like (e.g.)
        # TIMER_2019_03_31_at_05_50_12_IMG_0263
        has_no_datetime = df["datetime"].isna()
        fname_inner = df.loc[has_no_datetime, "image"].apply(
            lambda x: "_".join(x.split("_")[1:-2])
        )
        df.loc[has_no_datetime, "datetime"] = pd.to_datetime(
            fname_inner, format="%Y_%m_%d_at_%H_%M_%S"
        )
        if verbose >= 1:
            print(
                f"{ds_id}: Interpolating latitude, longitude, and depth for dataset {ds_id}"
            )
        df = interpolate_by_datetime(df, ["latitude", "longitude", "depth"])

    if ds_id in [914155]:
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
        # Images without datetime are too dark
        df = df[~df["datetime"].isna()]
        # Other images are missing latitude and longitude metadata
        df = interpolate_by_datetime(df, ["latitude", "longitude"])

    if ds_id in [914156, 914197]:
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
        # Some images are clearly of the same thing, but one is good visibility
        # with no lat/lon, and the next is too dark and has no datetime.
        for from_image, to_image in [
            ("IMG_0393", "IMG_0392"),
            ("IMG_0395", "IMG_0394"),
        ]:
            columns = ["latitude", "longitude"]
            select_from = df["image"].str.startswith(from_image)
            select_to = df["image"].str.startswith(to_image)
            df.loc[select_to, columns] = df.loc[select_from, columns]
        # Drop images without datetime
        df = df[~df["datetime"].isna()]
        # Fill in any missing latitude and longitude metadata
        df = interpolate_by_datetime(df, ["latitude", "longitude"])

    if ds_id in [914192]:
        if verbose >= 1:
            print(f"{ds_id}: Fixing missing metadata for dataset {ds_id}")
        # Some images are clearly of the same thing, but one is good visibility
        # with no lat/lon, and the next is too dark and has no datetime.
        for from_image, to_image in [
            ("IMG_1776", "IMG_1775"),
        ]:
            columns = ["latitude", "longitude"]
            select_from = df["image"].str.startswith(from_image)
            select_to = df["image"].str.startswith(to_image)
            df.loc[select_to, columns] = df.loc[select_from, columns]
        # Drop images without datetime
        df = df[~df["datetime"].isna()]
        # Fill in any missing latitude and longitude metadata
        df = interpolate_by_datetime(df, ["latitude", "longitude"])

    if any(df["latitude"].isna() | df["longitude"].isna()):
        # Fill in any missing latitude and longitude values with the
        # mean coordinate reported at the dataset level
        ds = PanDataSet(ds_id)
        if hasattr(ds, "geometryextent"):
            lat = None
            long = None
            for k in ["meanLatitude", "latitude", "Latitude"]:
                if k in ds.geometryextent:
                    lat = ds.geometryextent[k]
                    break
            for k in ["meanLongitude", "longitude", "Latitude"]:
                if k in ds.geometryextent:
                    long = ds.geometryextent[k]
                    break
            if lat is not None:
                if verbose >= 1:
                    print(f"{ds_id}: Using dataset mean latitude for missing values")
                df.loc[df["latitude"].isna(), "latitude"] = lat
            if long is not None:
                if verbose >= 1:
                    print(f"{ds_id}: Using dataset mean longitude for missing values")
                df.loc[df["longitude"].isna(), "longitude"] = long

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
        if ds_id == "805690":
            # The title was not captured from this dataset for some reason,
            # so we can't exclude it via the title.
            continue
        df = pd.read_csv(os.path.join(input_dirname, fname), low_memory=False)
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
        df = fixup_repeated_urls(df, url_column=url_col, verbose=verbose)

        if len(df) != len(df.drop_duplicates(subset=url_col)):
            files_with_repeat_urls2.append(fname)

        # Check for any rows that are all NaNs
        if sum(df.isna().all("columns")) > 0:
            print(f"{ds_id} has a row which is all NaNs")

        # Remove duplicated "favourited" images
        df = fixup_favourite_images(df, verbose=verbose)

        # Fix incomplete lat/lon/datetime metadata
        df = fixup_incomplete_metadata(df, ds_id, verbose=verbose)

        # Add datetime if it is completely missing
        df = add_missing_datetime(df, ds_id, verbose=verbose)

        dfs.append(df)
        dfs_fnames.append(fname)

    if verbose >= 0:
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
        df_all = fixup_repeated_output_paths(df_all, inplace=True, verbose=verbose)

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
    Run command line interface for cleaning and merging datasets.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return process_datasets(**kwargs)


if __name__ == "__main__":
    main()
