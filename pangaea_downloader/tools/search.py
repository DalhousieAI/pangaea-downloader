"""
Functions for searching Pangaea for benthic habitat images.
"""
from typing import List

from pangaeapy import PanQuery


def read_query_list(file="../pangaea_downloader/query_list") -> List[str]:
    """Read file with list of search queries and return it as a list."""
    with open(file, "r") as f:
        query_list = f.readlines()
    query_list = [query.strip() for query in query_list if query.strip() != ""]
    return query_list


def run_search_query(query: str, verbose=False, n_results=500) -> List[dict]:
    """Search Pangaea with given query string and return a list of results."""
    print(f"[INFO] Running search with query string: '{query}'") if verbose else 0
    offset = 0
    results = []
    # Iteratively retrieve search results
    while True:
        pq = PanQuery(query=query, limit=n_results, offset=offset)
        results.extend(pq.result)
        offset += len(pq.result)
        if len(results) >= pq.totalcount:
            break
    # Sanity check
    assert len(results) == pq.totalcount
    print(
        f"[INFO] Number of search results returned: {len(results)}\n"
    ) if verbose else 0
    return results


def run_multiple_search_queries(verbose=False) -> List[dict]:
    """Search Pangaea with multiple search queries and return a list of unique results."""
    # Read in list of search queries
    query_list = read_query_list()
    # Search multiple queries
    print("[INFO] Running multiple search queries...") if verbose else 0
    results_list = []
    for i, query in enumerate(query_list):
        search_results = run_search_query(query=query, n_results=500)
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
