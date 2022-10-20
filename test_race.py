import pytorch_lightning as pl
import torch
import numpy as np
import wandb
import pandas as pd
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast as BertTokenizer
from data.RaceDataModule import RaceDataModule
from model.Test import TEST
from model.model import BERT
from model.Duma import DUMA


if __name__ == "__main__":

    data_module = RaceDataModule(
        model_name="xlm-roberta-base",
        dataset_name="race",
        task_name="all",
        batch_size=16,
        max_seq_len=512,
        num_workers=4,
        num_proc=8,
        num_choices=4,
        version="flat",
    )

    data_module.setup()
