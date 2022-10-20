# CUDA_VISIBLE_DEVICES=0,1 python3 train_copa.py

from datasets import load_dataset
from transformers.adapters.composition import Stack

dataset_en = load_dataset("super_glue", "copa")
dataset_en.num_rows

dataset_en["train"].features

from transformers import AutoTokenizer

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def encode_batch(examples):
    """Encodes a batch of input data using the model tokenizer."""
    all_encoded = {"input_ids": [], "attention_mask": []}
    # Iterate through all examples in this batch
    for premise, question, choice1, choice2 in zip(
        examples["premise"],
        examples["question"],
        examples["choice1"],
        examples["choice2"],
    ):
        sentences_a = [premise + " " + question for _ in range(2)]
        # Both answer choices are passed in an array according to the format needed for the multiple-choice prediction head
        sentences_b = [choice1, choice2]
        encoded = tokenizer(
            sentences_a,
            sentences_b,
            max_length=60,
            truncation=True,
            padding="max_length",
        )
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
    return all_encoded


def preprocess_dataset(dataset):
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column("label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    return dataset


dataset_en = preprocess_dataset(dataset_en)

from transformers import AutoConfig, AutoAdapterModel

config = AutoConfig.from_pretrained(
    model_name,
)
model = AutoAdapterModel.from_pretrained(
    model_name,
    config=config,
)

from transformers import AdapterConfig

# Load the language adapters
lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
model.load_adapter("en/wiki@ukp", config=lang_adapter_config)
model.load_adapter("es/wiki@ukp", config=lang_adapter_config)

# Add a new task adapter
model.add_adapter("copa")

# Add a classification head for our target task
model.add_multiple_choice_head("copa", num_choices=2)

model.train_adapter(["copa"])

model.active_adapters = Stack("en", "copa")

from transformers import TrainingArguments, AdapterTrainer
from datasets import concatenate_datasets

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=8,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=100,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

train_dataset = concatenate_datasets([dataset_en["train"], dataset_en["validation"]])

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

dataset_zh = load_dataset("xcopa", "zh", ignore_verifications=True)
dataset_zh = preprocess_dataset(dataset_zh)

model.active_adapters = Stack("zh", "copa")

import numpy as np
from transformers import EvalPrediction


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


eval_trainer = AdapterTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="./eval_output",
        remove_unused_columns=False,
    ),
    eval_dataset=dataset_zh["test"],
    compute_metrics=compute_accuracy,
)
eval_trainer.evaluate()

model.save_pretrained("./run2/model")
model.save_all_adapters("./run2")
