import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.cluster import KMeans

H5_EMBEDDINGS_PATH = Path(__file__).parent.parent / "input_data"/"key_words.h5"
ENTITIES_PATH = Path(__file__).parent.parent/ "input_data" / "key_words.txt"

EMBEDDINGS_ENTITIES_PATH = Path(__file__).parent.parent / "output_data"/  "key_words.pt"
ENTITIES_CLUSTERS_PATH = Path(__file__).parent.parent / "output_data" / "key_words_clusters.json"
PRETTY_OUTPUT_PATH = Path(__file__).parent.parent / "output_data" /"key_words_clusters.txt"


def from_h5_to_pt(
    h5_path: str | Path, ent_path: str | Path, pt_path: str | Path
) -> None:
    ent_embed = dict()
    with h5py.File(h5_path, "r") as f:
        with open(ent_path, "r") as f_in:
            for i, input in enumerate(f_in):
                entity_name = input.strip()
                embedding = f[entity_name]["embedding"][:]
                ent_embed[entity_name] = embedding
    torch.save(ent_embed, pt_path)


def cluster_embeddings(
    ent_embed: dict[str, np.ndarray], n_clusters: int = 10
) -> dict[int, str]:
    embeddings = np.array(list(ent_embed.values()))
    entities = np.array(list(ent_embed.keys()))
    kmeans = KMeans(
        n_clusters=n_clusters,
    ).fit(embeddings)
    cluster_ent = dict()
    for entity, cluster in zip(entities, kmeans.labels_):
        cluster_ent.setdefault(int(cluster), []).append(
            entity
        )  # int(cluster) for correct types json serialization
    return cluster_ent


def make_pretty_output(
    cluster_ent: dict[int, str], output_path: str | Path, max_ent_per_cluster: int = 5
) -> None:
    text = ""
    for cluster, entities in cluster_ent.items():
        text = (
            text
            + f"CLUSTER {cluster} CONTAINS: "
            + "\n"
            + " ,".join(entities[:max_ent_per_cluster])
            + "\n"
        )
    Path(output_path).write_text(text)


def run_cluster_embedding_pipeline():
    ent_embed = torch.load(EMBEDDINGS_ENTITIES_PATH, weights_only=False)
    cluster_ent = cluster_embeddings(ent_embed=ent_embed)
    with open(ENTITIES_CLUSTERS_PATH, "w") as outfile:
        json.dump(cluster_ent, outfile)
    make_pretty_output(
        cluster_ent, output_path=PRETTY_OUTPUT_PATH, max_ent_per_cluster=5
    )


if __name__ == "__main__":
    from_h5_to_pt(
        H5_EMBEDDINGS_PATH, ENTITIES_PATH, EMBEDDINGS_ENTITIES_PATH
    )  # run for save embeddings from biobert in comfort format
    run_cluster_embedding_pipeline()  # run for save clusters with entities
