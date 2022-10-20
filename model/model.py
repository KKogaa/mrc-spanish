import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import (
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel,
)


class BERT(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bert = AutoModel.from_pretrained(kwargs["model_name"], return_dict=True)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1)

        self.learning_rate = kwargs["learning_rate"]
        self.batch_size = kwargs["batch_size"]
        self.num_choices = kwargs["num_choices"]
        self.criterion = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()

        self.positives = 0
        self.negatives = 0

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        x = bert_output.pooler_output

        output = self.fc1(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output.squeeze(-1), labels.float())
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.num_choices,
        )

        prob = torch.sigmoid(outputs)
        max_prediction = torch.argmax(prob.flatten())
        max_label = torch.argmax(labels)

        if torch.equal(max_prediction, max_label):
            self.positives = self.positives + 1
        else:
            self.negatives = self.negatives + 1

        # output = {"loss": loss, "prob": prob.flatten(), "target": labels.int()}

        return loss

    # def test_step_end(self, outputs):
    #     predictions = outputs["prob"].detach().cpu()
    #     targets = outputs["target"].detach().cpu()

    #     max_prediction = torch.argmax(predictions.flatten())
    #     max_label = torch.argmax(targets)

    #     if torch.equal(max_prediction, max_label):
    #         self.positives = self.positives + 1
    #     else:
    #         self.negatives = self.negatives + 1

    def test_epoch_end(self, outputs):
        accuracy = self.positives / (self.positives + self.negatives)
        self.log(f"test_accuracy_epoch", accuracy)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #   optimizer,
        #   num_warmup_steps=self.n_warmup_steps,
        #   num_training_steps=self.n_training_steps
        # )
        # return dict(
        #   optimizer=optimizer,
        #   lr_scheduler=dict(
        #     scheduler=scheduler,
        #     interval='step'
        #   )
        # )
        return optimizer
