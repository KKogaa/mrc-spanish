import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import regex as re
import nltk
from sklearn.manifold import TSNE
import plotly.express as px
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
    vectorizer = CountVectorizer(
        preprocessor=preprocess_text, ngram_range=ngram_range, stop_words=stop_words
    )
    vectorizer.fit_transform([text])
    return vectorizer.get_feature_names()


def get_embeddings(model, sentences):
    return model.encode(sentences, convert_to_tensor=True)


def compute_cosine_similarity(pivot, embeddings):

    similarities = []
    for embedding in list(embeddings):
        similarities.append(F.cosine_similarity(pivot, embedding, dim=0))

    return torch.stack(similarities)


def print_top_k(similarities, sentences, k):
    top_k = list(torch.topk(similarities, k, largest=True, sorted=True).indices)
    top_k = [tensor.item() for tensor in top_k]
    similarities = list(similarities)
    for idx in top_k:
        print(f"{sentences[idx]} {similarities[idx]}")

    return top_k


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


def print_similarities(similarities, sentences, max_idxs):
    for idx in max_idxs:
        print(f"{sentences[idx]} {similarities[idx]}")


if __name__ == "__main__":
    doc = """
    El profesor Manuel Tupia Anticona de la Especialidad de Ingeniería Informática participó en la 29th International Conference on Computer Applications
     in Industry and Engineering (CAINE 2016) celebrada entre el 26 y el 28 de setiembre en Denver (Colorado-USA) presentando el artículo 
     "An Information Security Framework for Services of M-Government from Peruvian Local Governments", en co-autoría con la alumna del Doctorado de Ingeniería, 
     Ing. Mariuxi Bruzza (Ecuador). La conferencia fue organizada por International Society for Computers and Their Applications (ISCA), la Universidad de Colorado. 
     El profesor Tupia fue igualmente session chair de la sesión relacionada a Seguridad Computacional. 
    """

    model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")

    sentences = ngram_text(doc, (1, 1))

    pivot_embedding = get_embeddings(model, doc)
    embeddings = get_embeddings(model, sentences)
    similarities = compute_cosine_similarity(pivot_embedding, embeddings)
    print("TOP K")
    top_k_idxs = print_top_k(similarities, sentences, 10)

    print("MAX FOR EACH CLUSTER")
    assigned_clusters = cluster_embeddings(embeddings, 10)
    groups = group_similarities(similarities, assigned_clusters, 10)
    max_idxs = find_max_for_each_group(groups)
    print_similarities(similarities, sentences, max_idxs)

    assigned_clusters.append(-1)

    ###################################################
    # VISUALIZATION
    ###################################################
    all_embeddings = list(embeddings)
    all_embeddings.append(pivot_embedding)
    all_embeddings = torch.stack(all_embeddings)
    sentences.append("DOCUMENT")

    labels = ["word" for _ in range(52)]
    labels.append("document")

    # for idx in top_k_idxs:
    #     labels[idx] = "top_k"

    for idx in max_idxs:
        labels[idx] = "top_cluster"

    clusters = [str(label) for label in assigned_clusters]

    X_embedded = TSNE(n_components=2).fit_transform(all_embeddings.cpu().numpy())
    # X_embedded = TSNE(n_components=2, perplexity=30, n_iter=500).fit_transform(all_embeddings.cpu().numpy())
    df_embeddings = pd.DataFrame(X_embedded)
    df_embeddings = df_embeddings.rename(columns={0: "x", 1: "y"})
    df_embeddings["text"] = sentences
    df_embeddings["label"] = labels
    df_embeddings["cluster"] = clusters

    fig = px.scatter(
        df_embeddings,
        x="x",
        y="y",
        color="label",
        text=df_embeddings["text"],
        labels={"color": "label"},
        hover_data=["cluster"],
        size_max=20,
    )
    fig.show()
