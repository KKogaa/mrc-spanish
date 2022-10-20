import pytorch_lightning as pl
import torch
import numpy as np
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast as BertTokenizer
from data.bert_datamodule import BERTDataModule
from model.model import BERT
import pandas as pd


def train(
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config
    ) as run:

        config = run.config

        model = BERT(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            model_name=config.model_name,
            num_choices=config.num_choices,
            num_epochs=config.epochs,
            gradient_accumulation_steps=config.accumulate,
            warmup_proportion=config.warmup_proportion,
        )

        logger = pl.loggers.WandbLogger(experiment=run)

        df_train = pd.read_csv("./data/translated/race_single_spanish_train.csv")
        df_val = pd.read_csv("./data/translated/race_single_spanish_val.csv")
        df_test = pd.read_csv("./data/translated/race_single_spanish_test.csv")

        data_module = BERTDataModule(
            model_name=config.model_name,
            batch_size=config.batch_size,
            max_seq_len=config.max_token_count,
            num_workers=4,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            num_choices=config.num_choices,
        )

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, monitor="val_accuracy_epoch", mode="min"
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_accuracy_epoch", patience=3
        )

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=[1, 2, 3],
            strategy="dp",
            logger=logger,
            accumulate_grad_batches=config.accumulate,
        )

        trainer.fit(model, data_module)

        torch.cuda.empty_cache()


def test(
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config
    ) as run:

        config = run.config
        logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        data_module = QuailDataModule(
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            task_name=config.task_name,
            batch_size=config.batch_size,
            max_seq_len=config.max_token_count,
            num_workers=4,
            num_proc=8,
            version="flat",
            num_choices=config.num_choices,
        )

        model = TEST.load_from_checkpoint(
            config.checkpoint,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            model_name=config.model_name,
            num_choices=config.num_choices,
            num_epochs=config.epochs,
            gradient_accumulation_steps=config.accumulate,
            warmup_proportion=config.warmup_proportion,
        )

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=[0, 1],
            strategy="dp",
            logger=logger,
        )

        trainer.test(model=model, datamodule=data_module)

        torch.cuda.empty_cache()


if __name__ == "__main__":

    project = "xlm-race-spanish"
    entity = None
    name = None
    sweep_config = {
        "method": "grid",  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        "metric": {  # This is the metric we are interested in minimizing or maximizing
            "name": "test_accuracy_epoch",
            "goal": "maximize",
        },
        # Paramters and parameter values we are sweeping across
        "parameters": {
            "learning_rate": {"values": [1e-5]},
            "batch_size": {"values": [16]},
            "epochs": {"values": [5]},
            "max_token_count": {"values": [400]},
            "model_name": {"values": ["xlm-roberta-base"]},
            "dataset_name": {"values": [None]},
            "task_name": {"values": [None]},
            "num_choices": {"values": [4]},
            "version": {"values": [None]},
            "accumulate": {"values": [1]},
            "warmup_proportion": {"values": [None]},
            "checkpoint": {"values": [None]},
        },
    }
    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
        entity=entity,
    )
    wandb.agent(sweep_id, function=train, count=1)
