# CUDA_VISIBLE_DEVICES=0,1 python3 train_copa.py

from datasets import load_dataset
from transformers.adapters.composition import Stack

dataset_en = load_dataset("quail")
dataset_en.num_rows

dataset_en["train"].features

from transformers import AutoTokenizer

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def encode_batch(examples):

    context = [[article] * 4 for article in examples["context"]]
    question_option = [
        [f"{question} {option}" for option in options]
        for options, question in zip(examples["answers"], examples["question"])
    ]

    context = sum(context, [])
    question_option = sum(question_option, [])

    encoding = tokenizer(
        context,
        question_option,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    encoding["input_ids"] = encoding["input_ids"].view(-1, 4, 512)
    encoding["attention_mask"] = encoding["attention_mask"].view(-1, 4, 512)
    labels = [answer for answer in examples["correct_answer_id"]]

    return {
        "input_ids": encoding["input_ids"].tolist(),
        "attention_mask": encoding["attention_mask"].tolist(),
        "labels": labels,
    }


def preprocess_dataset(dataset):
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
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

# Add a new task adapter
model.add_adapter("quail")

# Add a classification head for our target task
model.add_multiple_choice_head("quail", num_choices=4)

model.train_adapter(["quail"])

model.active_adapters = Stack("en", "quail")

from transformers import TrainingArguments, AdapterTrainer
from datasets import concatenate_datasets

training_args = TrainingArguments(
    learning_rate=1e-5,
    num_train_epochs=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
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
model.save_pretrained("./run_quail/model")
model.save_all_adapters("./run_quail")

# # import numpy as np
# # from transformers import EvalPrediction

# # def compute_accuracy(p: EvalPrediction):
# #   preds = np.argmax(p.predictions, axis=1)
# #   return {"acc": (preds == p.label_ids).mean()}

# # eval_trainer = AdapterTrainer(
# #     model=model,
# #     args=TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
# #     eval_dataset=dataset_en["challenge"],
# #     compute_metrics=compute_accuracy,
# # )
# # eval_trainer.evaluate()
