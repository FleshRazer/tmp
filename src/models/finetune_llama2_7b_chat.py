from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from datasets import DatasetDict
import wandb
import fire


def main(
        dataset_dir:
    str = "tmp/data/interim/NousResearch.Llama-2-7b-chat-hf_train_dataset",
        use_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_use_double_quant: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
        warmup_steps: int = 2,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-4,
        fp16: bool = False,
        bf16: bool = False,
        logging_steps: int = 5,
        output_dir: str = "tmp/models/llama2-7b-chat",
        optim: str = "paged_adamw_8bit",
        report_to: str = "none",
        evaluation_strategy: str = "no",
        eval_steps: int = 25,
        save_strategy: str = "no",
        save_steps: int = 1000):
    dataset = DatasetDict.load_from_disk(dataset_dir)

    model_name = "NousResearch/Llama-2-7b-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    if report_to == "wandb":
        wandb.init(project="Text Detoxification")

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=fp16,
            bf16=bf16,
            logging_steps=logging_steps,
            output_dir=output_dir,
            optim=optim,
            report_to=report_to,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.model.save_pretrained(f"{output_dir}/final")
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
