import pickle

import pandas as pd

from .tools import requesting


def get_bibtex(ds_id: str, verbose=False) -> str:
    """Get the BibTex Citation of a Pangaea dataset using the dataset ID."""
    bib_url = f"https://doi.pangaea.de/10.1594/PANGAEA.{ds_id}?format=citation_bibtex"
    resp = requesting.get_request_with_backoff(bib_url)
    if verbose:
        print("\tStatus code:", resp.status_code)
    return resp.text


def generate_citations_file(
    pangaea_dataset: str, citations_file: str, mappings_file: str
) -> None:
    """
    Generate a text file with BibTex citations for all Pangaea datasets in the `pangaea_dataset` CSV file.

    Parameters
    ----------
    pangaea_dataset : str
        The path to the CSV file containing the full Pangaea Benthic Image Dataset.
        It should contain a column called ``"datasets"`` with the dataset IDs.

    citations_file : str
        The path to the output bib file where all the citations will be written.

    mappings_file : str
        The path to the output pickle file to write the dataset ID to BibTex citation key mappings.
    """
    pangaea_df = pd.read_csv(pangaea_dataset, low_memory=False)
    ds_ids = [dataset.split("-")[-1] for dataset in pangaea_df.dataset.unique()]

    # Get citations
    citations = []
    mappings = {}
    print(f"[INFO] Processing {len(ds_ids)} dataset citations...")
    for i, ds_id in enumerate(ds_ids):
        # Get BibTex citation
        bibtex = get_bibtex(ds_id)
        citations.append(bibtex)
        # Extract BibTex tag
        tag = bibtex.split("{")[1].split(",")[0]
        mappings[int(ds_id)] = tag
        print(f"{(i + 1)}/{len(ds_ids)} complete.")

    # Write citations to file
    with open(citations_file, "w") as f:
        f.writelines(citations)
    print(f"[INFO] All dataset BibTex citations written to file: '{citations_file}'")
    # Write mappings to file
    pickle.dump(mappings, open(mappings_file, "wb"))
    print(
        f"[INFO] All dataset ID to BibTex tag mappings written to file: '{citations_file}'"
    )


if __name__ == "__main__":
    pangaea_file = "../full-dataset/pangaea_2022-01-02_filtered_subsampled-1.25m-1200-1000_remove-core-surf.csv"
    generate_citations_file(
        # Input file
        pangaea_file,
        # Generated output files
        "../pangaea-citations.bib",
        "../bibtex-key-id-mappings.pickle",
    )
