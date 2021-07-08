import os

import pangaeapy
from utilz import fetch_child_datasets, has_url_col


def fetch_query_data(query="seabed photographs", n_results=10, out_dir="query-outputs"):
    # Make output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Make a pangaea search query (returns max 500 results)
    pq = pangaeapy.PanQuery(query=query, limit=n_results)
    print("\n[INFO] Number of search results:", len(pq.result))

    # List of dataset DOIs
    result_dois = [r_item["URI"] for r_item in pq.result]

    # Process each result item
    print("[INFO] Processing each dataset...")
    for i, doi in enumerate(result_dois):
        # TODO: Check if dataset type (use size key of result dict)
        # Load dataset
        print(f"\n[{i+1}] Loading dataset DOI:'{doi}'...")
        try:
            ds = pangaeapy.PanDataSet(doi)
        except MemoryError:
            print("\t[ERROR] DATASET TOO LARGE!")
            continue
        # Get dataset ID
        ds_id = ds.doi.split(".")[
            -1
        ]  # TODO: ds_id can be used when we don't know the campaign name
        print(f"[INFO] Title: '{ds.title}'")

        # Dataset access restricted
        if ds.loginstatus != "unrestricted":
            print(f"\t[ERROR] Access restricted: '{ds.loginstatus}', DOI: {ds.doi}")
        else:  # Can access dataset
            if not ds.isParent:  # Does not have child datasets
                if has_url_col(ds.data):  # Has the desired url column
                    # Add metadata
                    ds.data["Dataset"] = ds.title
                    ds.data["DOI"] = ds.doi
                    ds.data["Campaign"] = ds.events[0].campaign.name  # ds_id?
                    ds.data["Site"] = ds.data["Event"]
                    # Save to file
                    file = os.path.join(out_dir, ds_id + ".csv")
                    ds.data.to_csv(file, index=False)
                    print(f"\t[INFO] Saved to '{file}'")
                else:
                    print(
                        f"\t[WARNING] Image URL column NOT FOUND! Data will NOT be saved! DOI: {ds.doi}"
                    )
            else:  # Dataset has child datasets
                df = fetch_child_datasets(ds)
                if df is None:
                    print("\t[ERROR] None of the child datasets had image URL column!")
                else:
                    # Save to file
                    file = os.path.join(out_dir, ds_id + ".csv")
                    df.to_csv(file, index=False)
                    print(f"\t[INFO] Saved to '{file}'")


if __name__ == "__main__":
    fetch_query_data(query="seabed photographs", n_results=999)
