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


def get_filtered_paranmt_tuning_hf(num_shards: int=4, test_size: float=0.25, seed: int=None):
    dataset = get_filtered_paranmt_hf()
    dataset = dataset.shard(num_shards=num_shards, index=0)

    prompt_template = (
        f"Detoxify the text, output only edited text:\n{{reference}}\n---\nDetoxified text:\n{{translation}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "sequence": prompt_template.format(
                reference=sample["reference"],
                translation=sample["translation"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: tokenizer(x["sequence"]), remove_columns=list(dataset.features), batched=True)
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset


def get_filtered_paranmt_inference_hf(num_shards: int=4, test_size: float=0.25, seed: int=None):
    dataset = get_filtered_paranmt_hf()
    dataset = dataset.shard(num_shards=num_shards, index=0)
    
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    dataset = dataset["test"]
    
    prompt_template = (
        f"Detoxify the text, output only edited text:\n{{reference}}\n---\nDetoxified text: "
    )
    
    def apply_prompt_template(sample):
        return {
            "sequence": prompt_template.format(
                reference=sample["reference"],
            )
        }
    
    remove_columns = list(dataset.features)
    remove_columns.remove("reference")
    remove_columns.remove("translation")
    dataset = dataset.map(apply_prompt_template, remove_columns=remove_columns)
    return dataset
