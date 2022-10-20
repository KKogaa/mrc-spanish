import pandas as pd
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
from ast import literal_eval
import ast


def translate(text, tokenizer, model):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    input_ids = (
        tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        .to(device)
        .input_ids
    )
    outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return decoded[0]


def translate_many(texts, tokenizer, model):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    t_many = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        outputs = model.generate(
            input_ids=input_ids, num_beams=5, num_return_sequences=1
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        t_many.append(decoded[0])

    return t_many


def translate_context(df_train, df_val, df_test, tokenizer, model):
    # df_val['translated_context'] = df_val.progress_apply(lambda x: translate(x.context, tokenizer, model), axis=1)
    # df_test['translated_context'] = df_test.progress_apply(lambda x: translate(x.context, tokenizer, model), axis=1)

    df_train["translated_context"] = df_train.progress_apply(
        lambda x: translate(x.context, tokenizer, model), axis=1
    )
    df_train["answers"] = df_train["answers"].map(lambda x: ",".join(map(str, x)))
    df_train["translated_answers"] = df_train["translated_answers"].map(
        lambda x: ",".join(map(str, x))
    )
    df_train.to_csv("./data/translated_quail_train.csv", index=False)


def translate_answers(df_train, df_val, df_test, tokenizer, model):
    df_train["translated_answers"] = df_train.progress_apply(
        lambda x: translate_many(x.answers, tokenizer, model), axis=1
    )
    df_val["translated_answers"] = df_val.progress_apply(
        lambda x: translate_many(x.answers, tokenizer, model), axis=1
    )
    df_test["translated_answers"] = df_test.progress_apply(
        lambda x: translate_many(x.answers, tokenizer, model), axis=1
    )


def translate_question(df_train, df_val, df_test, tokenizer, model):
    df_train["translated_question"] = df_train.progress_apply(
        lambda x: translate(x.question, tokenizer, model), axis=1
    )
    df_val["translated_question"] = df_val.progress_apply(
        lambda x: translate(x.question, tokenizer, model), axis=1
    )
    df_test["translated_question"] = df_test.progress_apply(
        lambda x: translate(x.question, tokenizer, model), axis=1
    )


tqdm.pandas()

if __name__ == "__main__":
    # train dev test
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # dataset = datasets.load_dataset("quail", None)
    # df_train = dataset["train"].to_pandas()
    # df_val = dataset["validation"].to_pandas()
    # df_test = dataset["challenge"].to_pandas()
    # df_train = pd.read_pickle("data/quail_translated_train")
    # df_val = pd.read_pickle("data/quail_translated_val")
    # df_test = pd.read_pickle("data/quail_translated_test")
    # df_val.to_csv("./data/translated_quail_val.csv", index=False)
    # df_test.to_csv("./data/translated_quail_test.csv", index=False)

    # df_train.to_pickle("./data/p4_quail_translated_train", protocol=4)
    # df_val.to_pickle("./data/p4_quail_translated_val", protocol=4)
    # df_test.to_pickle("./data/p4_quail_translated_test", protocol=4)

    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    # model = model.to(device)

    # authenticator = IAMAuthenticator('OMRtmH7YvNMYuLQCfc1PK0PuRo0TeZuW22tlhmArjxCw')
    # language_translator = LanguageTranslatorV3(
    #     version='2018-05-01',
    #     authenticator=authenticator
    # )

    # language_translator.set_service_url('https://api.us-south.language-translator.watson.cloud.ibm.com/instances/f12a7b48-4c34-4a12-9e60-6783dde8ed55')

    df_train = pd.read_csv("./data/translated_quail_train.csv")
    print(len(df_train))

    # translate_question(df_train, df_val, df_test, tokenizer, model)
    # translate_context(df_train, df_val, df_test, tokenizer, model)
    # translate_answers(df_train, df_val, df_test, tokenizer, model)

    # df_train['answers'] = df_train['answers'].map(lambda x: ','.join(map(str, x)))
    # df_val['answers'] = df_val['answers'].map(lambda x: ','.join(map(str, x)))
    # df_test['answers'] = df_test['answers'].map(lambda x: ','.join(map(str, x)))

    # df_train['translated_answers'] = df_train['translated_answers'].map(lambda x: ','.join(map(str, x)))
    # df_val['translated_answers'] = df_val['translated_answers'].map(lambda x: ','.join(map(str, x)))
    # df_test['translated_answers'] = df_test['translated_answers'].map(lambda x: ','.join(map(str, x)))

    # df_train.to_csv("./data/translated_quail_train.csv", index=False)
    # df_val.to_csv("./data/translated_quail_val.csv", index=False)
    # df_test.to_csv("./data/translated_quail_test.csv", index=False)
