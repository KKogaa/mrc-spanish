import pandas as pd
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
from ast import literal_eval
import ast

if __name__ == "__main__":
    # train dev test
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    # model = model.to(device)

    dataset = datasets.load_dataset("quail", None)
    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()
    df_test = dataset["challenge"].to_pandas()
