import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import conceptnet_lite
from conceptnet_lite import Label
import numpy as np
from operator import itemgetter
import requests
from nltk.cluster import KMeansClusterer


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    return text

def ngram_text(text, ngram_range):
    stopword_es = nltk.corpus.stopwords.words("spanish")
    stopword_en = nltk.corpus.stopwords.words("english")
    stop_words = stopword_en + stopword_es
    stop_words = frozenset(stop_words)

    try:
        vectorizer = CountVectorizer(
            preprocessor=preprocess_text, ngram_range=ngram_range, stop_words=stop_words
        )
        vectorizer.fit_transform([text])
        return vectorizer.get_feature_names()
    except Exception as e:
        print(e)
        return None


def compute_cosine_similarity(pivot, embeddings):
    similarities = []
    for embedding in list(embeddings):
        similarities.append(F.cosine_similarity(pivot, embedding, dim=0))

    return torch.stack(similarities)


def find_top_k(similarities, sentences, k):
    top_k = list(torch.topk(similarities, k, largest=True, sorted=True).indices)
    top_k = [tensor.item() for tensor in top_k]
    similarities = list(similarities)

    return top_k


def calculate_concept(source, sentences, model):
    if sentences is None:
        return None

    pivot_embedding = get_embeddings(model, source)
    embeddings = get_embeddings(model, sentences)
    similarities = compute_cosine_similarity(pivot_embedding, embeddings)
    top_idx = find_top_k(similarities, sentences, 1)
    return sentences[top_idx[0]]


def identify_key_concepts_dataframes(
    df_train,
    df_val,
    df_test,
    pivot_column_name,
    target_column_name,
    ngram_column_name,
    concept_column_name,
    csv_output_file_name,
):

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    df_train[ngram_column_name] = df_train.progress_apply(
        lambda x: ngram_text(x[target_column_name], (1, 1)), axis=1
    )
    df_train[concept_column_name] = df_train.progress_apply(
        lambda x: calculate_concept(x[pivot_column_name], x[ngram_column_name], model),
        axis=1,
    )
    df_train.to_csv(f"data/enhanced/{csv_output_file_name}_train.csv", index=False)

    df_val[ngram_column_name] = df_val.progress_apply(
        lambda x: ngram_text(x[target_column_name], (1, 1)), axis=1
    )
    df_val[concept_column_name] = df_val.progress_apply(
        lambda x: calculate_concept(x[pivot_column_name], x[ngram_column_name], model),
        axis=1,
    )
    df_val.to_csv(f"data/enhanced/{csv_output_file_name}_val.csv", index=False)

    df_test[ngram_column_name] = df_test.progress_apply(
        lambda x: ngram_text(x[target_column_name], (1, 1)), axis=1
    )
    df_test[concept_column_name] = df_test.progress_apply(
        lambda x: calculate_concept(x[pivot_column_name], x[ngram_column_name], model),
        axis=1,
    )
    df_test.to_csv(f"data/enhanced/{csv_output_file_name}_test.csv", index=False)


def consult_wikidata(text):
    r = requests.get(
        f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={text}&language=en&format=json"
    ).json()
    datas = []
    for term in r["search"]:
        data = term.get("label", " ") + ". " + term.get("description", " ") + "."
        datas.append(data)
    return datas


def wikidata(pivot, text, model):
    datas = consult_wikidata(text)
    if datas:
        embeddings = get_embeddings(model, datas)
        pivot_embedding = get_embeddings(model, pivot)
        sim = compute_cosine_similarity(pivot_embedding, embeddings)
        idx = find_top_k(sim, datas, 1)
        return datas[idx[0]]
    return None

def get_embeddings(model, sentences):
    return model.encode(sentences, convert_to_tensor=True)

def enhance_data(
    df_train,
    df_val,
    df_test,
    knowledge_column_name,
    pivot_column_name,
    concept_column_name,
    csv_output_file_name,
):

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    df_train[knowledge_column_name] = df_train.progress_apply(
        lambda x: wikidata(x[pivot_column_name], x[concept_column_name], model), axis=1
    )
    df_train.to_csv(f"data/enhanced/{csv_output_file_name}_train.csv", index=False)

    df_val[knowledge_column_name] = df_val.progress_apply(
        lambda x: wikidata(x[pivot_column_name], x[concept_column_name], model), axis=1
    )
    df_val.to_csv(f"data/enhanced/{csv_output_file_name}_val.csv", index=False)

    df_test[knowledge_column_name] = df_test.progress_apply(
        lambda x: wikidata(x[pivot_column_name], x[concept_column_name], model), axis=1
    )
    df_test.to_csv(f"data/enhanced/{csv_output_file_name}_test.csv", index=False)


def translate(text, tokenizer, model):
    if text is None:
        return None

    try:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        input_ids = (
            tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            .to(device)
            .input_ids
        )
        outputs = model.generate(
            input_ids=input_ids, num_beams=5, num_return_sequences=1
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decoded[0]
    except:
        return None


def translate_data(
    df_train,
    df_val,
    df_test,
    target_column_name,
    translated_column_name,
    csv_output_file_name,
):

    device = "cuda:1" if torch.cuda.is_available() else "cpu" 
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    model = model.to(device)

    df_train[translated_column_name] = df_train.progress_apply(
        lambda x: translate(x[target_column_name], tokenizer, model), axis=1
    )
    df_train.to_csv(f"data/enhanced/{csv_output_file_name}_train.csv", index=False)

    df_val[translated_column_name] = df_val.progress_apply(
        lambda x: translate(x[target_column_name], tokenizer, model), axis=1
    )
    df_val.to_csv(f"data/enhanced/{csv_output_file_name}_val.csv", index=False)

    df_test[translated_column_name] = df_test.progress_apply(
        lambda x: translate(x[target_column_name], tokenizer, model), axis=1
    )
    df_test.to_csv(f"data/enhanced/{csv_output_file_name}_test.csv", index=False)


tqdm.pandas()

if __name__ == "__main__":

    df_train = pd.read_csv("data/recores_train.csv")
    df_val = pd.read_csv("data/recores_val.csv")
    df_test = pd.read_csv("data/recores_test.csv")


    print("IDENTIFY KEY CONCEPTS")
    identify_key_concepts_dataframes(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        pivot_column_name="text",
        target_column_name="question",
        ngram_column_name="ngram_question",
        concept_column_name="question_concept",
        csv_output_file_name="recores_key_question",
    )

    identify_key_concepts_dataframes(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        pivot_column_name="text",
        target_column_name="answer",
        ngram_column_name="ngram_answer",
        concept_column_name="answer_concept",
        csv_output_file_name="recores_key_answer",
    )

    print("TRANSLATE CONCEPTS")
    translate_data(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        target_column_name="question_concept",
        translated_column_name="translated_question_concept",
        csv_output_file_name="recores_key_translated_question",
    )

    translate_data(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        target_column_name="answer_concept",
        translated_column_name="translated_answer_concept",
        csv_output_file_name="recores_key_translated_answer",
    )


    print("ENHANCE CONCEPTS")
    enhance_data(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        knowledge_column_name="question_knowledge",
        pivot_column_name="text",
        concept_column_name="question_concept",
        csv_output_file_name="recores_key_knowledge_question",
    )

    enhance_data(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        knowledge_column_name="answer_knowledge",
        pivot_column_name="text",
        concept_column_name="answer_concept",
        csv_output_file_name="recores_key_knowledge_answer",
    )

    print("FINAL RESULT")

    df_train.to_csv(f"data/enhanced/final_train.csv", index=False)
    df_val.to_csv(f"data/enhanced/final_val.csv", index=False)
    df_test.to_csv(f"data/enhanced/final_test.csv", index=False)

