import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel,
)


def separate_seq2(sequence_output, flat_input_ids):
    qa_seq_output = sequence_output.new(sequence_output.size()).zero_()
    qa_mask = torch.ones(
        (sequence_output.shape[0], sequence_output.shape[1]),
        device=sequence_output.device,
        dtype=torch.bool,
    )
    p_seq_output = sequence_output.new(sequence_output.size()).zero_()
    p_mask = torch.ones(
        (sequence_output.shape[0], sequence_output.shape[1]),
        device=sequence_output.device,
        dtype=torch.bool,
    )
    for i in range(flat_input_ids.size(0)):
        # 0   1 2 3 4   5 6 7 8
        # cls a b c sep x y z sep
        # sep_idx = 4
        # 1:3
        # 5:7
        sep_lst = []
        for idx, e in enumerate(flat_input_ids[i]):
            if e == 102:
                sep_lst.append(idx)
        assert len(sep_lst) == 2
        qa_seq_output[i, : sep_lst[0] - 1] = sequence_output[i, 1 : sep_lst[0]]
        qa_mask[i, : sep_lst[0] - 1] = 0
        p_seq_output[i, : sep_lst[1] - sep_lst[0] - 1] = sequence_output[
            i, sep_lst[0] + 1 : sep_lst[1]
        ]
        p_mask[i, : sep_lst[1] - sep_lst[0] - 1] = 0
    return qa_seq_output, p_seq_output, qa_mask, p_mask


class DUMALayer(nn.Module):
    def __init__(self, d_model_size, num_heads):
        super(DUMALayer, self).__init__()
        self.attn_qa = nn.MultiheadAttention(d_model_size, num_heads)
        self.attn_p = nn.MultiheadAttention(d_model_size, num_heads)

    def forward(
        self, qa_seq_representation, p_seq_representation, qa_mask=None, p_mask=None
    ):
        qa_seq_representation = qa_seq_representation.permute([1, 0, 2])
        p_seq_representation = p_seq_representation.permute([1, 0, 2])
        enc_output_qa, _ = self.attn_qa(
            value=qa_seq_representation,
            key=qa_seq_representation,
            query=p_seq_representation,
            key_padding_mask=qa_mask,
        )
        enc_output_p, _ = self.attn_p(
            value=p_seq_representation,
            key=p_seq_representation,
            query=qa_seq_representation,
            key_padding_mask=p_mask,
        )
        return enc_output_qa.permute([1, 0, 2]), enc_output_p.permute([1, 0, 2])


class DUMA(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bert = AutoModel.from_pretrained(kwargs["model_name"], return_dict=True)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.duma = DUMALayer(
            self.bert.config.hidden_size, self.bert.config.num_attention_heads
        )

        self.learning_rate = kwargs["learning_rate"]
        self.batch_size = kwargs["batch_size"]
        self.num_choices = kwargs["num_choices"]
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        self.num_epochs = kwargs["num_epochs"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.warmup_proportion = kwargs["warmup_proportion"]
        self.warmup_steps = 0
        self.total_steps = 0

    def setup(self, stage):
        if stage == "fit":
            train_loader = (
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )
            self.total_steps = int(
                len(train_loader.dataset)
                / self.batch_size
                / self.gradient_accumulation_steps
                * self.num_epochs
            )
            self.warmup_steps = int(self.total_steps * self.warmup_proportion)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        last_output = bert_output.last_hidden_state
        qa_seq_output, p_seq_output, qa_mask, p_mask = separate_seq2(
            last_output, input_ids
        )
        enc_output_qa, enc_output_p = self.duma(
            qa_seq_output, p_seq_output, qa_mask, p_mask
        )
        fused_output = torch.cat([enc_output_qa, enc_output_p], dim=1)
        pooled_output = torch.mean(fused_output, dim=1)
        logits = self.fc1(self.dropout(pooled_output))
        reshaped_logits = logits.view(-1, self.num_choices)
        loss = 0
        if labels is not None:
            loss = self.criterion(reshaped_logits, labels)
        return loss, reshaped_logits

    def training_step(self, batch, batch_idx):
        # unflatten inputids and attention mask (batch_size, 2048) -> (batch_size * 4, 512)
        input_ids = batch["input_ids"].view(
            batch["input_ids"].shape[0] * self.num_choices, -1
        )
        attention_mask = batch["attention_mask"].view(
            batch["attention_mask"].shape[0] * self.num_choices, -1
        )
        labels = batch["label"]

        loss, logits = self(input_ids, attention_mask, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

        probs = F.softmax(logits, dim=1)
        labels_hat = torch.argmax(probs, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(
            batch["input_ids"].shape[0] * self.num_choices, -1
        )
        attention_mask = batch["attention_mask"].view(
            batch["attention_mask"].shape[0] * self.num_choices, -1
        )
        labels = batch["label"]

        loss, logits = self(input_ids, attention_mask, labels)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

        probs = F.softmax(logits, dim=1)
        labels_hat = torch.argmax(probs, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        return {
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch["label"]),
        }

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].view(
            batch["input_ids"].shape[0] * self.num_choices, -1
        )
        attention_mask = batch["attention_mask"].view(
            batch["attention_mask"].shape[0] * self.num_choices, -1
        )
        labels = batch["label"]

        loss, logits = self(input_ids, attention_mask, labels)

        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

        probs = F.softmax(logits, dim=1)
        labels_hat = torch.argmax(probs, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        return {
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch["label"]),
        }

    def validation_epoch_end(self, outputs):
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(
            out["batch_size"] for out in outputs
        )
        self.log(f"val_accuracy_epoch", val_acc)

    def test_epoch_end(self, outputs):
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(
            out["batch_size"] for out in outputs
        )
        self.log(f"test_accuracy_epoch", test_acc)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #   optimizer,
        #   num_warmup_steps=self.warmup_steps,
        #   num_training_steps=self.total_steps
        # )
        # return dict(
        #   optimizer=optimizer,
        #   lr_scheduler=dict(
        #     scheduler=scheduler,
        #     interval='step'
        #   )
        # )
        return optimizer
