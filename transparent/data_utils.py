from transformers.data import default_data_collator
from transformers import AutoTokenizer
import torch
from datasets import load_dataset


block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def get_tokenized_wikitext():

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    datasets = load_dataset('wikitext', 'wikitext-103-raw-v1')

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=512, truncation=True)
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    tokenizer.decode(lm_datasets["train"][1]["input_ids"])

    train_loader = torch.utils.data.DataLoader(
        lm_datasets["train"], batch_size=64, shuffle=True, collate_fn=default_data_collator)
    validation_loader = torch.utils.data.DataLoader(
        lm_datasets["validation"], batch_size=32, shuffle=True, collate_fn=default_data_collator)
    return train_loader, validation_loader, tokenizer
