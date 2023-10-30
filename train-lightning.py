from datasets import Dataset
import pandas as pd
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler

block_size = 128
num_epochs = 5
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tokenize_function(examples):
    return tokenizer(
        examples["summary"], max_length=100, padding="max_length", truncation=True
    )


def group_texts(examples):
    # Concatenate all texts.
    # print(examples)
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


class CausalLMModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss

    def generate(self, inputs):
        return self.model.generate(inputs)


def main():
    # Read in data from book summaries dataset
    summaries = pd.read_csv("booksummaries/booksummaries.txt", sep="\t")
    summaries.columns = [
        "wikipedia_id",
        "freebase_id",
        "title",
        "author",
        "publication_date",
        "genres",
        "summary",
    ]

    dataset = Dataset.from_pandas(summaries[["summary"]])

    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["summary"]
    )

    # Define dataset prepped for causal language modeling
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=10,
        num_proc=4,
    )

    lm_dataset.set_format("torch")

    # Create dataloader with collate_fn (shifted by one to the right for causal language modeling)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        lm_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
    )

    model = CausalLMModel()

    trainer = Trainer(max_epochs=1, devices=1, accelerator="gpu")
    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()
