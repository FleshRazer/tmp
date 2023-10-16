from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)
from peft import PeftModel
from datasets import DatasetDict
import fire


def postprocess_generated_text_hf(generated_text: str):
    generated_text = generated_text.split("Detoxified text:")[-1].strip()
    generated_text = generated_text.split("\n")[0].strip()
    return generated_text


def main(
    use_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_use_double_quant: bool = True,
    adapter_dir: str = "tmp/models/llama2-7b-chat/checkpoint-2000",
    dataset_dir:
    str = "tmp/data/interim/NousResearch.Llama-2-7b-chat-hf_inference_dataset",
    output_filename:
    str = "tmp/data/interim/NousResearch.Llama-2-7b-chat-hf_inference_output.txt"
):
    dataset = DatasetDict.load_from_disk(dataset_dir)

    model_name = "NousResearch/Llama-2-7b-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto")

    model = PeftModel.from_pretrained(model, adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=250)

    pipe_output = pipe(dataset["sequence"])
    pipe_output = [
        postprocess_generated_text_hf(entry[0]["generated_text"])
        for entry in pipe_output
    ]

    return pipe_output


if __name__ == "__main__":
    fire.Fire(main)
