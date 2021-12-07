"""
Functions for searching Pangaea for benthic habitat images.
"""
import os
from inspect import getsourcefile
from typing import List

from pangaeapy import PanQuery

from .checker import check_allclose_dict

# Determine the path to the directory containing this file
TOOLS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
# Determine path to parent directory (package directory)
PACKAGE_DIRECTORY = os.path.dirname(TOOLS_DIRECTORY)


def read_query_list(file=None) -> List[str]:
    """Read file with list of search queries and return it as a list."""
    if file is None:
        file = os.path.join(PACKAGE_DIRECTORY, "query_list")
    with open(file, "r") as f:
        query_list = f.readlines()
    query_list = [query.strip() for query in query_list if query.strip() != ""]
    return query_list


def run_search_query(query: str, verbose=0, n_results=500) -> List[dict]:
    """Search Pangaea with given query string and return a list of results."""
    if verbose >= 1:
        print(f"[INFO] Running search with query string: '{query}'")
    offset = 0
    results = []
    last_top_result = {}
    n_page = 0
    # Iteratively retrieve search results
    while True:
        pq = PanQuery(query=query, limit=n_results, offset=offset)
        n_page += 1
        if verbose >= 2:
            print(f"\tResults page {n_page} (offset = {offset})")
        if len(pq.result) == 0 or check_allclose_dict(
            pq.result[0], last_top_result, exclude_keys=("position")
        ):
            # No more results to add
            break
        last_top_result = pq.result[0]
        results.extend(pq.result)
        offset += len(pq.result)
        if len(results) >= pq.totalcount:
            break
    # Sanity check
    if len(results) != pq.totalcount:
        print(
            "[WARNING]"
            f" Mismatch between total number of results ({len(results)}) and"
            f" expected number of results ({pq.totalcount})"
            f" for search '{query}'"
        )
    if verbose >= 1:
        print(f"[INFO] Number of search results returned: {len(results)}\n")
    return results


def run_multiple_search_queries(query_list, verbose=0) -> List[dict]:
    """Search Pangaea with multiple search queries and return a list of unique results."""
    # Search multiple queries
    if verbose >= 1:
        ps = f"[INFO] Running {len(query_list)} search queries"
        ps += ":" if verbose >= 2 else "..."
        if verbose >= 2:
            for query in query_list:
                ps += f"\n    {query}"
        print(ps)
    results_list = []
    for i, query in enumerate(query_list):
        search_results = run_search_query(query=query, verbose=verbose - 1)
        if verbose:
            print(
                f"\t[{i+1}] query: '{query}', results returned: {len(search_results)}"
            )
        results_list.extend(search_results)
    # Keep only unique results
    results_set = list({value["URI"]: value for value in results_list}.values())
    if verbose:
        print(f"[INFO] Number of unique search results: {len(results_set)}\n")
    return results_set
