import pytorch_lightning as pl
import wandb
import torch
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast as BertTokenizer

from data.recores_data import RecoresData
from data.bert_datamodule import BERTDataModule
from model.model import BERT


def train(
    project,
    entity,
    name,
    max_token_count,
    tokenizer,
    model_name,
    df_train,
    df_val,
    df_test,
    config={"learning_rate": 1e-5, "batch_size": 16, "epochs": 10},
):
    with wandb.init(
        project=project, entity=entity, job_type="train", config=config, name=name
    ) as run:

        # Extract the config object associated with the run
        config = run.config

        # Construct our LightningModule with the learning rate from the config object
        model = BERT(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            model_name=model_name,
        )

        # This logger is used when we call self.log inside the LightningModule
        # name_string = f"{config.batch_size}-{config.learning_rate}"
        logger = pl.loggers.WandbLogger(experiment=run, log_model=True)

        # build data module
        data_module = BERTDataModule(
            df_train,
            df_val,
            df_test,
            tokenizer,
            batch_size=config.batch_size,
            max_token_len=max_token_count,
        )

        # Construct a Trainer object with the W&B logger we created and epoch set by the config object
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, verbose=True, monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)

        trainer = pl.Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=config.epochs,
            gpus=[2, 3],
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

    dataset = RecoresData(
        sep="\t",
        train="train_spanish.csv",
        val="dev_spanish.csv",
        test="test_spanish.csv",
    )

    dataset.setup()

    (df_train, df_val, df_test) = dataset.get_dataframes()

    # define tokenizer
    model_name = "dccuchile/bert-base-spanish-wwm-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # train model
    project = "mrc_test"
    entity = None
    name = "test"
    config = {"learning_rate": 1e-5, "batch_size": 16, "epochs": 10}
    train(
        project=project,
        entity=entity,
        name=name,
        max_token_count=512,
        tokenizer=tokenizer,
        model_name=model_name,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        config=config,
    )
