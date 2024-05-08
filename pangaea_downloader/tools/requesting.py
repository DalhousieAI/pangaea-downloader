"""
URL request utilities.
"""

import time

import requests


def get_request_with_backoff(url, retries=5, backoff_factor=1, verbose=1, **kwargs):
    """
    Fetch a URL resource using requests with a custom backoff strategy for re-attempts.

    Parameters
    ----------
    url : str
        The URL to request.
    retries : int, default=5
        Maximum number of attempts.
    backoff_factor : float, default=1
        Base time to wait for before attempting to download again when receiving
        a 500 or 503 HTTP status code.
    verbose : int, default=1
        Verbosity level.
    **kwargs
        Additional arguments as per :func:`requests.get`.
    """
    for i_attempt in range(retries):
        r = requests.get(url, **kwargs)
        if r.status_code not in [429, 500, 503]:
            # Status code looks good
            break
        # N.B. Could also retry on [408, 502, 504, 599]
        if r.status_code == 429:
            # PANGAEA has a maximum of 180 requests within a 30s period
            # Wait for this to cool off completely.
            t_wait = 30
        else:
            # Other errors indicate a server side error. Wait a
            # short period and then retry to see if it alleviates.
            t_wait = backoff_factor * 2**i_attempt
        if verbose >= 1:
            print(
                "Retrying in {} seconds (HTTP Status {}): {}".format(
                    t_wait, r.status_code, url
                )
            )
        time.sleep(t_wait)
    return r
