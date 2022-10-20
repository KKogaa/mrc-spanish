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
    except:
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


def identify_key_concepts_dataframes(df_train, df_val, df_test, model):
    # ngram question
    df_train["ngram_question"] = df_train.progress_apply(
        lambda x: ngram_text(x.question, (1, 1)), axis=1
    )
    df_val["ngram_question"] = df_val.progress_apply(
        lambda x: ngram_text(x.question, (1, 1)), axis=1
    )
    df_test["ngram_question"] = df_test.progress_apply(
        lambda x: ngram_text(x.question, (1, 1)), axis=1
    )

    # question concept
    df_train["question_concept"] = df_train.progress_apply(
        lambda x: calculate_concept(x.text, x.ngram_question, model), axis=1
    )
    df_val["question_concept"] = df_val.progress_apply(
        lambda x: calculate_concept(x.text, x.ngram_question, model), axis=1
    )
    df_test["question_concept"] = df_test.progress_apply(
        lambda x: calculate_concept(x.text, x.ngram_question, model), axis=1
    )

    # ngram answers
    df_train["ngram_answer"] = df_train.progress_apply(
        lambda x: ngram_text(x.answer, (1, 1)), axis=1
    )
    df_val["ngram_answer"] = df_val.progress_apply(
        lambda x: ngram_text(x.answer, (1, 1)), axis=1
    )
    df_test["ngram_answer"] = df_test.progress_apply(
        lambda x: ngram_text(x.answer, (1, 1)), axis=1
    )

    # answer concept
    df_train["answer_concept"] = df_train.progress_apply(
        lambda x: calculate_concept(x.text, x.ngram_answer, model), axis=1
    )
    df_val["answer_concept"] = df_val.progress_apply(
        lambda x: calculate_concept(x.text, x.ngram_answer, model), axis=1
    )
    df_test["answer_concept"] = df_test.progress_apply(
        lambda x: calculate_concept(x.text, x.ngram_answer, model), axis=1
    )


# def identify_key_concepts_context(df_train, df_val, df_test, model):


def translate_key_concepts_dataframes(df_train, df_val, df_test, tokenizer, model):
    df_train["question_concept"] = df_train.progress_apply(
        lambda x: translate(x.question_concept, tokenizer, model), axis=1
    )
    df_val["question_concept"] = df_val.progress_apply(
        lambda x: translate(x.question_concept, tokenizer, model), axis=1
    )
    df_test["question_concept"] = df_test.progress_apply(
        lambda x: translate(x.question_concept, tokenizer, model), axis=1
    )

    df_train["answer_concept"] = df_train.progress_apply(
        lambda x: translate(x.answer_concept, tokenizer, model), axis=1
    )
    df_val["answer_concept"] = df_val.progress_apply(
        lambda x: translate(x.answer_concept, tokenizer, model), axis=1
    )
    df_test["answer_concept"] = df_test.progress_apply(
        lambda x: translate(x.answer_concept, tokenizer, model), axis=1
    )


def translate(text, tokenizer, model):

    if text is None:
        return None

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
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


def get_embeddings(model, sentences):
    return model.encode(sentences, convert_to_tensor=True)


def find_edges_out(word, target):
    concepts = Label.get(text=word, language="en").concepts
    neighbours = []
    for c in concepts:
        if c.edges_out:
            for e in c.edges_out:
                if str(e.end.language) == "en":
                    data = {
                        "end": e.end.text,
                        "relation": e.relation.name,
                        "weight": e.etc["weight"],
                    }
                    neighbours.append(data)

                if str(e.end.text) == target:
                    return {
                        "end": e.end.text,
                        "relation": e.relation.name,
                        "weight": e.etc["weight"],
                    }, None

    return None, neighbours


def match_conceptnet(word, target):
    target, relations = find_edges_out("author", "universalist")
    if target is None:
        for relation in relations:
            second_target, second_relations = find_edges_out(
                relation["end"], "universalist"
            )
            if second_target != None:
                print(f"{relation} {second_target}")

    else:
        print(f"{target}")


def get_top_k_conceptnet(word, k):
    conceptnet_lite.connect("./conceptnet/conceptnet.db", db_download_url=None)
    concepts = Label.get(text=word, language="en").concepts
    neighbours = []
    weights = []
    for c in concepts:
        if c.edges_out:
            for e in c.edges_out:
                if str(e.end.language) == "en":
                    data = {
                        "end": e.end.text,
                        "relation": e.relation.name,
                        "weight": e.etc["weight"],
                    }
                    neighbours.append(data)
                    weights.append(e.etc["weight"])

    top_ks = list(sorted(enumerate(weights), key=itemgetter(1)))[-k:]
    return [neighbours[idx] for idx, val in top_ks]


def explore_conceptnet_dataframes(df_train, df_val, df_test):
    conceptnet_lite.connect("./conceptnet/conceptnet.db", db_download_url=None)
    # df_train['conceptnet'] = df_train.progress_apply(lambda x: match_conceptnet(x.question_concept, x.answer_concept), axis=1)
    # df_val['conceptnet'] = df_val.progress_apply(lambda x: match_conceptnet(x.question_concept, x.answer_concept), axis=1)
    df_test["conceptnet"] = df_test.progress_apply(
        lambda x: match_conceptnet(x.question_concept, x.answer_concept), axis=1
    )


def consult_wikidata(text):
    r = requests.get(
        f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={text}&language=en&format=json"
    ).json()
    datas = []
    # print(len(r["search"]))
    for term in r["search"]:
        # print(term)
        data = term.get("label", " ") + ". " + term.get("description", " ") + "."
        datas.append(data)
    return datas


def wikidata(pivot, text, model):
    datas = consult_wikidata(text)

    if datas:
        try:
            embeddings = get_embeddings(model, datas)
            pivot_embedding = get_embeddings(model, pivot)
            sim = compute_cosine_similarity(pivot_embedding, embeddings)
            idx = find_top_k(sim, datas, 1)
            return datas[idx[0]]
        except:
            return None
    return None


def extract_data_from_concept(df_train, df_val, df_test, model):
    # df_train['question_concept_data'] = df_train.progress_apply(lambda x: wikidata(x.text, x.question_concept, model), axis=1)
    # df_val['question_concept_data'] = df_val.progress_apply(lambda x: wikidata(x.text, x.question_concept, model), axis=1)
    df_test["question_concept_data"] = df_test.progress_apply(
        lambda x: wikidata(x.text, x.question_concept, model), axis=1
    )

    # df_train['answer_concept_data'] = df_train.progress_apply(lambda x: wikidata(x.text, x.answer_concept, model), axis=1)
    # df_val['answer_concept_data'] = df_val.progress_apply(lambda x: wikidata(x.text, x.answer_concept, model), axis=1)
    df_test["answer_concept_data"] = df_test.progress_apply(
        lambda x: wikidata(x.text, x.answer_concept, model), axis=1
    )


def cluster_embeddings(embeddings, num_clusters):

    X = embeddings.cpu().numpy()

    kclusterer = KMeansClusterer(
        num_clusters,
        distance=nltk.cluster.util.cosine_distance,
        repeats=25,
        avoid_empty_clusters=True,
    )

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    return assigned_clusters


def group_similarities(similarities, assigned_clusters, num_clusters):
    groups = [{"similarities": [], "indexes": []} for _ in range(num_clusters)]
    for idx, cluster_number in enumerate(assigned_clusters):
        groups[cluster_number]["similarities"].append(similarities[idx])
        groups[cluster_number]["indexes"].append(idx)

    return groups


def find_max_for_each_group(groups):
    max_similarities = []
    for group in groups:
        temp = torch.stack(group["similarities"])
        max_idx = torch.argmax(temp).cpu().numpy()
        max_similarities.append(group["indexes"][max_idx])

    return max_similarities


tqdm.pandas()

if __name__ == "__main__":
    # df_train = pd.read_csv("data/recores_train.csv")
    # df_val = pd.read_csv("data/recores_val.csv")
    # df_test = pd.read_csv("data/recores_test.csv")

    # model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # identify_key_concepts_dataframes(df_train, df_val, df_test, model)

    # df_train.to_csv("data/temp_1_train.csv")
    # df_val.to_csv("data/temp_1_val.csv")
    # df_test.to_csv("data/temp_1_test.csv")

    # df_train = pd.read_csv("data/temp_1_train.csv")
    # df_val = pd.read_csv("data/temp_1_val.csv")
    # df_test = pd.read_csv("data/temp_1_test.csv")

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    # model = model.to(device)

    # translate_key_concepts_dataframes(df_train, df_val, df_test, tokenizer, model)

    # df_train.to_csv("data/temp_2_train.csv")
    # df_val.to_csv("data/temp_2_val.csv")
    # df_test.to_csv("data/temp_2_test.csv")

    # df_train = pd.read_csv("data/temp_2_train.csv")
    # df_val = pd.read_csv("data/temp_2_val.csv")
    # df_test = pd.read_csv("data/temp_2_test.csv")

    # extract_data_from_concept(df_train, df_val, df_test, model)

    # print(df_test)
    # df_train.to_csv("data/temp_2_train.csv")
    # df_val.to_csv("data/temp_2_val.csv")
    # df_test.to_csv("data/temp_3_test.csv")

    # explore_conceptnet_dataframes(df_train, df_val, df_test)

    # print(df_test)
    # print(get_top_k_conceptnet("car", 10))

    # consult_wikidata("yellow")
