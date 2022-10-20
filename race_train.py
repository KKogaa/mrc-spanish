import pytorch_lightning as pl
import torch
import numpy as np
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast as BertTokenizer
from data.RaceDataModule import RaceDataModule
from model.Test import TEST
from model.model import BERT


def train_race(
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config, name=name
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

        logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        data_module = RaceDataModule(
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            task_name=config.task_name,
            batch_size=config.batch_size,
            max_seq_len=config.max_token_count,
            num_workers=4,
            num_proc=8,
            version=config.version,
            num_choices=config.num_choices,
        )

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=4,
            strategy="dp",
            logger=logger,
        )

        trainer.fit(model, data_module)

        trainer.test(model=model, datamodule=data_module, ckpt_path="best")

        torch.cuda.empty_cache()


if __name__ == "__main__":

    project = "race_v1"
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
            "epochs": {"values": [20]},
            "max_token_count": {"values": [512]},
            "model_name": {"values": ["xlm-roberta-base"]},
            "dataset_name": {"values": ["race"]},
            "task_name": {"values": ["all"]},
            "num_choices": {"values": [4]},
            "version": {"values": ["flat"]},
            "accumulate": {"values": [1]},
            "warmup_proportion": {"values": [0.1]},
            "checkpoint": {"values": [None]},
        },
    }
    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
        entity=entity,
    )
    wandb.agent(sweep_id, function=train_race, count=1)
