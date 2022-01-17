import pandas as pd
import requests


def get_bibtex(ds_id: str) -> str:
    """Get the BibTex Citation of a Pangaea dataset using the dataset ID."""
    bib_url = f"https://doi.pangaea.de/10.1594/PANGAEA.{ds_id}?format=citation_bibtex"
    resp = requests.get(bib_url)
    return resp.text


def generate_citations_file(pangaea_dataset: str, citations_file: str) -> None:
    """
    Generate a text file with BibTex citations for all Pangaea datasets in the `pangaea_dataset` CSV file.

    Parameters
    ----------
    pangaea_dataset : str
        The path to the CSV file containing the full Pangaea Benthic Image Dataset.
        It should contain a column called ``"datasets"`` with the dataset IDs.

    citations_file : str
        The path to the output text file where all the citations will be written.
    """
    pangaea_df = pd.read_csv(pangaea_dataset, low_memory=False)
    ds_ids = [dataset.split("-")[-1] for dataset in pangaea_df.dataset.unique()]

    # Get bibtex citations and write to file
    print(f"[INFO] Processing {len(ds_ids)} dataset citations...")
    with open(citations_file, "w") as f:
        for i, ds_id in enumerate(ds_ids):
            bibtex = get_bibtex(ds_id)
            f.write(bibtex)
            print(f"\t{str(i+1).zfill(len(str(len(ds_ids))))}/{len(ds_ids)} complete.")
    print(f"[INFO] All dataset BibTex citations written to file: '{citations_file}'")


if __name__ == "__main__":
    file = (
        "../pangaea_2022-01-02_filtered_subsampled-1.25m-1200-1000_remove-core-surf.csv"
    )
    generate_citations_file(file, citations_file="../pangaea-citations.bib")
