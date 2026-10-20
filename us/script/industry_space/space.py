from __future__ import annotations
import itertools
import hashlib
import random
from typing import Any
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

from ..settings import ALPHA, N_CLUSTERS, SEED



NODE2VEC_PARAMS = {"dimensions": 128, "walk_length": 30, "num_walks": 200,
                   "p": 1, "q": 1, "weight_key": "weight", "workers": 1, "seed": 42}
WORD2VEC_PARAMS = {"window": 5, "epochs": 5, "negative": 5,
                   "min_count": 1, "sg": 1, "workers": 1, "seed": 42}

def stable_hash(value: Any) -> int:
    """Stable Gensim vocabulary hash independent of PYTHONHASHSEED."""
    return int.from_bytes(
        hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "little"
    )

def run_node2vec_embedding(
    backbone: nx.Graph,
    seed: int = 42,
    node2vec_overrides: dict[str, Any] | None = None,
    word2vec_overrides: dict[str, Any] | None = None,
):
    """Train deterministic single-worker node2vec and return model/vectors/order."""
    from node2vec import Node2Vec

    random.seed(seed)
    np.random.seed(seed)
    node2vec_params = dict(NODE2VEC_PARAMS)
    node2vec_params["seed"] = seed
    if node2vec_overrides:
        node2vec_params.update(node2vec_overrides)
    word2vec_params = dict(WORD2VEC_PARAMS)
    word2vec_params["seed"] = seed
    if word2vec_overrides:
        word2vec_params.update(word2vec_overrides)

    trainer = Node2Vec(backbone, quiet=True, **node2vec_params)
    model = trainer.fit(hashfxn=stable_hash, **word2vec_params)
    node_order = sorted(backbone.nodes)
    embeddings = np.vstack([model.wv[str(node)] for node in node_order]).astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("node2vec produced non-finite embeddings")
    return model, embeddings, node_order


def proximity(rca: np.ndarray) -> np.ndarray:
    binary = (rca > 1).astype(np.float64)
    joint = binary.T @ binary
    counts = np.diag(joint)
    left = np.divide(joint, counts[:, None], out=np.zeros_like(joint), where=counts[:, None] > 0)
    right = np.divide(joint, counts[None, :], out=np.zeros_like(joint), where=counts[None, :] > 0)
    phi = np.minimum(left, right)
    np.fill_diagonal(phi, 0)
    assert np.allclose(phi, phi.T) and np.isfinite(phi).all()
    return phi


def rca(participations, industry_universe=None):
    if participations.empty:
        return pd.DataFrame(columns=industry_universe or [], dtype=float)
    counts = participations.groupby(["investor_id", "naics4"]).size().unstack(fill_value=0)
    if industry_universe is not None:
        counts = counts.reindex(columns=industry_universe, fill_value=0)
    sector_share = counts.sum(axis=0) / counts.values.sum()
    investor_share = counts.div(counts.sum(axis=1), axis=0)
    return investor_share.div(sector_share.replace(0, np.nan), axis=1).fillna(0)


def relatedness(rca_matrix):
    return pd.DataFrame(proximity(rca_matrix.to_numpy()),
                        index=rca_matrix.columns, columns=rca_matrix.columns)


def weighted_network(phi):
    graph = nx.Graph()
    graph.add_nodes_from(phi.index)
    for i, j in itertools.combinations(phi.index, 2):
        weight = float(phi.at[i, j])
        if weight > 0:
            graph.add_edge(i, j, weight=weight)
    return graph


def disparity_backbone(graph, alpha=ALPHA):
    kept = nx.Graph()
    kept.add_nodes_from(graph.nodes)
    strengths = dict(graph.degree(weight="weight"))
    degrees = dict(graph.degree())
    for i, j, data in graph.edges(data=True):
        weight = data["weight"]
        ai = (1 - weight / strengths[i]) ** (degrees[i] - 1) if degrees[i] > 1 else 0.0
        aj = (1 - weight / strengths[j]) ** (degrees[j] - 1) if degrees[j] > 1 else 0.0
        if ai < alpha or aj < alpha:
            kept.add_edge(i, j, weight=weight)
    kept.remove_nodes_from(list(nx.isolates(kept)))
    return kept


def embed_and_cluster(graph):
    _, embedded, nodes = run_node2vec_embedding(graph, seed=SEED)
    vectors = pd.DataFrame(embedded, index=nodes)
    labels = pd.Series(KMeans(n_clusters=N_CLUSTERS, random_state=SEED,
                              n_init=20).fit_predict(vectors), index=nodes, name="cluster")
    return vectors, labels


def plot_space(graph, vectors, labels, destination, title):
    xy = umap.UMAP(n_components=2, random_state=SEED).fit_transform(vectors)
    plt.figure(figsize=(9, 7))
    positions = dict(zip(vectors.index, xy))
    for i, j in graph.edges:
        first, second = positions[i], positions[j]
        plt.plot([first[0], second[0]], [first[1], second[1]],
                 color="0.8", linewidth=.35, alpha=.45, zorder=1)
    sizes = [30 + 8 * graph.degree(node) for node in vectors.index]
    plt.scatter(xy[:, 0], xy[:, 1], c=labels.loc[vectors.index], s=sizes,
                cmap="tab10", alpha=.85, zorder=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(destination, dpi=300)
    plt.close()


def construct_space(participations):
    matrix = rca(participations)
    phi = relatedness(matrix)
    graph = disparity_backbone(weighted_network(phi))
    vectors, labels = embed_and_cluster(graph)
    return matrix, phi, graph, vectors, labels


def save_space(kind, participations):
    """Compute one industry space and save matrices, edges, clusters, and figure."""
    from ..settings import OUT_SPACE
    matrix, phi, graph, vectors, labels = construct_space(participations)
    matrix.to_csv(OUT_SPACE / f"{kind}_rca.csv")
    phi.to_csv(OUT_SPACE / f"{kind}_relatedness.csv")
    vectors.to_csv(OUT_SPACE / f"{kind}_embedding.csv")
    labels.to_csv(OUT_SPACE / f"{kind}_clusters.csv", header=True)
    pd.DataFrame([(i, j, d["weight"]) for i, j, d in graph.edges(data=True)],
                 columns=["naics_i", "naics_j", "weight"]).to_csv(
        OUT_SPACE / f"{kind}_backbone_edges.csv", index=False)
    plot_space(graph, vectors, labels, OUT_SPACE / f"{kind}_industry_space.png",
               "US" + (" lead" if kind == "lead" else ""))
    return matrix, phi, graph, vectors, labels


def load_space(kind="full"):
    """Read a previously saved industry space from this country's result directory."""
    from ..settings import OUT_SPACE
    matrix = pd.read_csv(OUT_SPACE / f"{kind}_rca.csv", index_col=0, dtype={"naics4": str})
    phi = pd.read_csv(OUT_SPACE / f"{kind}_relatedness.csv", index_col=0)
    vectors = pd.read_csv(OUT_SPACE / f"{kind}_embedding.csv", index_col=0)
    labels = pd.read_csv(OUT_SPACE / f"{kind}_clusters.csv", index_col=0).iloc[:, 0]
    for frame in (matrix, phi, vectors):
        frame.index = frame.index.map(str)
    matrix.columns = matrix.columns.map(str)
    phi.columns = phi.columns.map(str)
    labels.index = labels.index.map(str)
    edges = pd.read_csv(OUT_SPACE / f"{kind}_backbone_edges.csv",
                        dtype={"naics_i": str, "naics_j": str})
    graph = nx.Graph()
    graph.add_nodes_from(vectors.index)
    graph.add_weighted_edges_from(edges.itertuples(index=False, name=None))
    return matrix, phi, graph, vectors, labels


def main():
    """Construct and plot the full and lead-only industry spaces."""
    from ..settings import OUT_PREP, OUT_SPACE
    OUT_SPACE.mkdir(parents=True, exist_ok=True)
    mapped = pd.read_csv(OUT_PREP / "industry_participations.csv",
                         dtype={"naics4": str}, low_memory=False)
    save_space("full", mapped)
    lead = mapped.loc[mapped.is_lead.astype(str).str.lower().eq("true")]
    if lead.empty:
        raise ValueError("No lead investments are available")
    save_space("lead", lead)
    print(f"Industry spaces saved: full={len(mapped)}, lead={len(lead)} participations")


if __name__ == "__main__":
    main()
