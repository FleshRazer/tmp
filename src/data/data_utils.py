import os

import requests
import zipfile

from pandas import read_csv
from datasets import Dataset


filtered_paranmt_archive_filename = "tmp/data/raw/filtered_paranmt.zip"
filtered_paranmt_tsv_filename = "tmp/data/raw/filtered.tsv"


def download_filtered_paranmt():
    url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
    response = requests.get(url, stream=True)

    os.makedirs(os.path.dirname(filtered_paranmt_archive_filename), exist_ok=True)
    with open(filtered_paranmt_archive_filename, mode="wb") as _file:
        for chunk in response.iter_content(chunk_size=128):
            _file.write(chunk)


def get_filtered_paranmt_df():
    return read_csv("tmp/data/raw/filtered.tsv", sep="\t")


def get_filtered_paranmt_hf():
    df = get_filtered_paranmt_df()
    df = df[df["reference"].str.len() <= 250]
    dataset = Dataset.from_dict({column: df.loc[:, column].to_list() for column in df.columns})

    return dataset
