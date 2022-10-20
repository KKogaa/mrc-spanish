import pandas as pd
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
from ast import literal_eval
import ast
import nltk


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

    texts = [str(text) for text in texts]
    input_ids = (
        tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        .to(device)
        .input_ids
    )
    outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return decoded

    # return t_many


nltk.download("punkt")


def split_context_sentences(context):
    sentences = nltk.tokenize.sent_tokenize(context)
    return sentences


def join_sentences(context):
    sentences = " ".join(context)
    return sentences


def translate_context(df_train, df_val, df_test, tokenizer, model):
    df_train["split_context"] = df_train.progress_apply(
        lambda x: split_context_sentences(x.article), axis=1
    )
    # df_val['split_context'] = df_val.progress_apply(lambda x: split_context_sentences(x.article), axis=1)
    # df_test['split_context'] = df_test.progress_apply(lambda x: split_context_sentences(x.article), axis=1)

    # df_val['translated_context'] = df_val.progress_apply(lambda x: translate_many(x.split_context, tokenizer, model), axis=1)
    # df_val['translated_context'] = df_val.progress_apply(lambda x: join_sentences(x.translated_context), axis=1)
    # df_val.to_csv("./data/translated_race_val_3.csv", index=False)

    # df_test['translated_context'] = df_test.progress_apply(lambda x: translate_many(x.split_context, tokenizer, model), axis=1)
    # df_test['translated_context'] = df_test.progress_apply(lambda x: join_sentences(x.translated_context), axis=1)
    # df_test.to_csv("./data/translated_race_test_3.csv", index=False)

    df_train["translated_context"] = df_train.progress_apply(
        lambda x: translate_many(x.split_context, tokenizer, model), axis=1
    )
    df_train["translated_context"] = df_train.progress_apply(
        lambda x: join_sentences(x.translated_context), axis=1
    )
    df_train.to_csv("./data/translated_race_train_3.csv", index=False)


def translate_answers(df_train, df_val, df_test, tokenizer, model):
    # df_test['translated_answers'] = df_test.progress_apply(lambda x: translate_many(x.options, tokenizer, model), axis=1)
    # df_test['translated_answers'] = df_test.progress_apply(lambda x: remove_comma(x.options), axis=1)
    # df_test['translated_answers'] = df_test['translated_answers'].map(lambda x: ','.join(map(str, x)))
    # df_test.to_csv("./data/translated_race_test_2.csv", index=False)

    # df_train['translated_answers'] = df_train.progress_apply(lambda x: translate_many(x.options, tokenizer, model), axis=1)
    # df_train['translated_answers'] = df_train.progress_apply(lambda x: remove_comma(x.options), axis=1)
    # df_train['translated_answers'] = df_train['translated_answers'].map(lambda x: ','.join(map(str, x)))
    # df_train.to_csv("./data/translated_race_train_2.csv", index=False)

    df_val["translated_answers"] = df_val.progress_apply(
        lambda x: translate_many(x.options, tokenizer, model), axis=1
    )
    df_val["translated_answers"] = df_val.progress_apply(
        lambda x: remove_comma(x.options), axis=1
    )
    df_val["translated_answers"] = df_val["translated_answers"].map(
        lambda x: ",".join(map(str, x))
    )
    df_val.to_csv("./data/translated_race_val_2.csv", index=False)


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


def remove_comma(texts):
    new_texts = []
    for text in texts:
        t = text.replace(",", " ")
        new_texts.append(t)

    return new_texts


tqdm.pandas()

if __name__ == "__main__":
    # train dev test
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = model.to(device)

    dataset = datasets.load_dataset("race", "all")
    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()
    df_test = dataset["test"].to_pandas()

    # df_train = df_train.drop_duplicates(subset=['example_id'])
    # df_val = df_val.drop_duplicates(subset=['example_id'])
    # df_test = df_test.drop_duplicates(subset=['example_id'])

    # df_train = pd.read_csv("data/translated_race_train_2.csv")
    # df_val = pd.read_csv("data/translated_race_val_2.csv")
    # df_test = pd.read_csv("data/translated_race_test_2.csv")

    translate_answers(df_train, df_val, df_test, tokenizer, model)

    # print(df_train.head())

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

    # df_train = pd.read_csv("./data/translated_quail_train.csv")
    # print(len(df_train))

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
