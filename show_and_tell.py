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


def train(
    project,
    entity,
    name,
    max_token_count,
    model_name,
    dataset_name,
    task_name,
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config, name=name
    ) as run:

        # Extract the config object associated with the run
        config = run.config

        # Construct our LightningModule with the learning rate from the config object
        model = TEST(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            model_name=model_name,
        )

        # This logger is used when we call self.log inside the LightningModule
        # name_string = f"{config.batch_size}-{config.learning_rate}"
        logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        # build data module
        data_module = RaceDataModule(
            model_name=model_name,
            dataset_name=dataset_name,
            task_name=task_name,
            batch_size=config.batch_size,
            max_seq_len=max_token_count,
            num_workers=4,
            num_proc=8,
        )

        # Construct a Trainer object with the W&B logger we created and epoch set by the config object
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

        # Execute training
        trainer.fit(model, data_module)

        # load best model
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")

        # "/drive/MyDrive/qa_datasets_spanish/spanish_mcqa_v1/4ford7jt/checkpoints/epoch=7-step=2615.ckpt"
        # model = BERT.load_from_checkpoint(
        #     "/drive/MyDrive/qa_datasets_spanish/spanish_mcqa_v1/4ford7jt/checkpoints/epoch=7-step=2615.ckpt",
        #     learning_rate=1e-5,
        # )

        torch.cuda.empty_cache()


if __name__ == "__main__":

    project = "race_test_4"
    entity = None
    learning_rate = 1e-5
    batch_size = 8
    name = f"test-lr-{learning_rate}-bs-{batch_size}"
    config = {"learning_rate": learning_rate, "batch_size": batch_size, "epochs": 5}
    test(
        project=project,
        entity=entity,
        name=name,
        max_token_count=512,
        model_name="roberta-base",
        dataset_name="race",
        task_name="all",
        num_choices=4,
        config=config,
    )
