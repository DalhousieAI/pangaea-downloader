"""
Contains helpful functions for checking various conditions.
"""

import re

from pandas import DataFrame

VALID_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# --------------------------------------------- String Checkers --------------------------------------------- #
def is_url(string: str) -> bool:
    """
    Check if input string is a valid URL or not.

    src: https://stackoverflow.com/questions/7160737/how-to-validate-a-url-in-python-malformed-or-not
    """
    if not isinstance(string, str):
        return False
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, string) is not None


def has_img_extension(filename: str) -> bool:
    """Check if a filename has a valid image file extension."""
    # Must be string
    if not isinstance(filename, str):
        return False
    # File should have valid image file extension
    return filename.lower().endswith(VALID_IMG_EXTENSIONS)


def is_img_url(url: str) -> bool:
    """Check if a given URL has a valid image file extension."""
    # Must be string and valid url
    if not (isinstance(url, str) and is_url(url)):
        return False
    # Should end with valid image file extension
    return url.lower().endswith(VALID_IMG_EXTENSIONS)


def is_invalid_file_ext(filename: str) -> bool:
    """Check if file has unwanted file extension."""
    if not isinstance(filename, str):
        return False
    # TODO: read list of invalid file extensions from file
    exclude_files = [".pdf", ".docx", ".doc", ".pptx", ".key", ".xlsx", ".mov", ".mp4"]
    ext = "." + filename.split(".")[-1].lower()
    if ext in exclude_files:
        return True
    return False


# --------------------------------------------- DataFrame Checkers --------------------------------------------- #
def has_url_col(df: DataFrame) -> bool:
    """Take a Pandas DataFrame and return True if it has image URL column."""
    condition1 = any(["url" in col.lower() for col in df.columns])
    condition2 = any(["image" in col.lower() for col in df.columns])
    return condition1 or condition2
