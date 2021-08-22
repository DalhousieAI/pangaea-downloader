[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Pangaea Downloader

This repository contains tools to search and download benthic habitat image datasets from [Pangaea](https://www.pangaea.de/).

![Sample Images](./images/sample_images.png)

## Installation

You can download the repository using git and install the package with pip.

```bash
git clone git@github.com:DalhousieAI/pangaea-downloader.git
cd pangaea-downloader
pip install .
```

## Run downloader

Run `pangaea_downloader/pq_scraper.py` to download benthic habitat image datasets from Pangaea.

The downloaded dataset files will be stored in the main project directory under `query-outputs`.


## Developmental setup

To develop the project, some additional set up steps are needed.

The repository can be cloned and installed (in editable mode) as follows:

```bash
git clone git@github.com:DalhousieAI/pangaea-downloader.git
cd pangaea-downloader
pip install -e .[dev]
```

The repository uses the [black](https://black.readthedocs.io/) code style, and uses [pre-commit](https://pre-commit.com/) for quality assurance during development.

Pre-commit automatically sanitizes your code as you commit it.
To install pre-commit and set up the pre-commit hooks, run the following code:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

From then on, whenever you try to commit code which needs to be modified by a commit hook, you'll have to add the changes made by pre-commit and then redo your commit.

You can also manually run the pre-commit stack on all the files at any time:

```bash
pre-commit run --all-files
```
