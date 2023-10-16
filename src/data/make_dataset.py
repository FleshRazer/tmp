import fire

from pandas import read_csv
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


def _make_llama2_train_dataset(df_train, df_test, tokenizer):
    dataset_train = Dataset.from_pandas(df_train, split="train")
    dataset_test = Dataset.from_pandas(df_test, split="train")

    prompt_template = f"Detoxify the text, output only edited text:\n{{reference}}\n---\nDetoxified text:\n{{translation}}{{eos_token}}"

    def apply_prompt_template(sample):
        return {
            "sequence":
            prompt_template.format(
                reference=sample["reference"],
                translation=sample["translation"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset_train = dataset_train.map(apply_prompt_template,
                                      remove_columns=list(
                                          dataset_train.features))
    dataset_train = dataset_train.map(lambda x: tokenizer(x["sequence"]),
                                      remove_columns=list(
                                          dataset_train.features),
                                      batched=True)

    dataset_test = dataset_test.map(apply_prompt_template,
                                    remove_columns=list(dataset_test.features))
    dataset_test = dataset_test.map(lambda x: tokenizer(x["sequence"]),
                                    remove_columns=list(dataset_test.features),
                                    batched=True)

    dataset = DatasetDict()
    dataset["train"] = dataset_train
    dataset["test"] = dataset_test
    return dataset


def _make_llama2_inference_dataset(df_train, df_test):
    dataset_train = Dataset.from_pandas(df_train, split="train")
    dataset_test = Dataset.from_pandas(df_test, split="train")

    prompt_template = f"Detoxify the text, output only edited text:\n{{reference}}\n---\nDetoxified text: "

    def apply_prompt_template(sample):
        return {
            "sequence": prompt_template.format(reference=sample["reference"], )
        }

    remove_columns = list(dataset_train.features)
    remove_columns.remove("reference")
    remove_columns.remove("translation")

    dataset_train = dataset_train.map(apply_prompt_template,
                                      remove_columns=remove_columns)
    dataset_test = dataset_test.map(apply_prompt_template,
                                    remove_columns=remove_columns)

    dataset = DatasetDict()
    dataset["train"] = dataset_train
    dataset["test"] = dataset_test
    return dataset


def main(dataset_filename: str,
         model_name: str,
         sample_frac: float = None,
         max_reference_len: int = None,
         test_size: float = 0.25,
         seed: int = None):
    df = read_csv(dataset_filename, sep="\t")

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=seed)
    if max_reference_len is not None:
        df = df[df["reference"].str.len() <= max_reference_len]

    df_train, df_test = train_test_split(df,
                                         test_size=test_size,
                                         random_state=seed)

    if model_name == "NousResearch/Llama-2-7b-chat-hf":
        train_dataset_filename = f"tmp/data/interim/{model_name.replace('/', '.')}_train_dataset"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = _make_llama2_train_dataset(df_train, df_test,
                                                   tokenizer)
        train_dataset.save_to_disk(train_dataset_filename)
        print(train_dataset)
        print("\nSaved to", train_dataset_filename)

        inference_dataset_filename = f"tmp/data/interim/{model_name.replace('/', '.')}_inference_dataset"
        inference_dataset = _make_llama2_inference_dataset(df_train, df_test)
        inference_dataset.save_to_disk(inference_dataset_filename)
        print(inference_dataset)
        print("\nSaved to", inference_dataset_filename)
        return

    if model_name == "":
        pass


if __name__ == "__main__":
    fire.Fire(main)
