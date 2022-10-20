import conceptnet_lite
from conceptnet_lite import Label
import requests


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


def query_conceptnet(word):
    obj = requests.get(
        f"http://api.conceptnet.io/c/en/{word}", params={"limit": 1000}
    ).json()

    relations = []
    for edge in obj["edges"]:
        if edge["end"]["label"] != word:
            data = {
                "surfaceText": edge["surfaceText"],
                "weight": edge["weight"],
                "rel": edge["rel"]["label"],
                "end": edge["end"]["label"],
            }
            relations.append(data)

    return relations


def query_conceptnet_target(word, target):
    obj = requests.get(
        f"http://api.conceptnet.io/c/en/{word}", params={"limit": 1000}
    ).json()

    relations = []
    for edge in obj["edges"]:
        if edge["end"]["label"] != word:
            data = {
                "surfaceText": edge["surfaceText"],
                "weight": edge["weight"],
                "rel": edge["rel"]["label"],
                "end": edge["end"]["label"],
            }
            relations.append(data)
        if target in edge["end"]["label"]:
            return {
                "surfaceText": edge["surfaceText"],
                "weight": edge["weight"],
                "rel": edge["rel"]["label"],
                "end": edge["end"]["label"],
            }, None

    return None, relations


def query_wiki_data(word):
    pass


def query_dbpedia(word):
    pass


if __name__ == "__main__":
    conceptnet_lite.connect("./conceptnet/conceptnet.db", db_download_url=None)

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
