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


def get_tokenized_wikitext(args):

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

    train_loader = torch.utils.data.DataLoader(
        lm_datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)
    validation_loader = torch.utils.data.DataLoader(
        lm_datasets["validation"], batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)
    return train_loader, validation_loader, tokenizer
    
def get_tokenized_openwebtext(args, tokenizer):

    datasets = load_dataset('stas/openwebtext-10k', split='train[:10%]')

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

    data_loader = torch.utils.data.DataLoader(
        lm_datasets, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)
    return data_loader

def get_tokenized_code(args, tokenizer):

    code_data = load_dataset("codeparrot/codeparrot-valid-v2-near-dedup", split="train[:1%]")
    for key in code_data.features:
        if key != "content":
            code_data = code_data.remove_columns(key)
    def tokenize_function(examples):
        return tokenizer(examples["content"], max_length=512, truncation=True)
    tokenized_datasets = code_data.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["content"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    data_loader = torch.utils.data.DataLoader(
        lm_datasets, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)
    return data_loader

