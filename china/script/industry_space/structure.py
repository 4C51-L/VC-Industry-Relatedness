from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
import umap

from ..settings import SEED, SUPPLEMENT
from .space import weighted_network, disparity_backbone, embed_and_cluster


def within_cluster_cosines(vectors, labels):
    similarity = cosine_similarity(vectors)
    rows = []
    nodes = list(vectors.index)
    for a, b in itertools.combinations(range(len(nodes)), 2):
        rows.append((nodes[a], nodes[b], similarity[a, b],
                     labels.loc[nodes[a]] == labels.loc[nodes[b]]))
    return pd.DataFrame(rows, columns=["industry_i", "industry_j", "cosine", "same_cluster"])


def rank_historical_changes(vectors, country):
    history = pd.read_csv(SUPPLEMENT / "naics_revision_relations.csv", dtype={"naics4": str})
    history = history.loc[history.country.eq(country) & history.naics4.isin(vectors.index)]
    rows = []
    for code in history.naics4.unique():
        subsector = code[:3]
        members = [x for x in vectors.index if x.startswith(subsector)]
        centroid = vectors.loc[members].mean(axis=0).to_numpy()
        rows.append((code, subsector, cosine(vectors.loc[code].to_numpy(), centroid)))
    return pd.DataFrame(rows, columns=["naics4", "subsector", "centroid_distance"]).sort_values(
        "centroid_distance", ascending=False)


def plot_internal_similarity(pairs, ranked, country, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pairs.loc[pairs.same_cluster, "cosine"], bins=40, alpha=.75)
    axes[0].set(title=f"{country}: within-cluster cosine", xlabel="Cosine similarity")
    selected = ranked.head(10).sort_values("centroid_distance")
    threshold = ranked.centroid_distance.quantile(.8)
    colors = ["crimson" if x >= threshold else "steelblue" for x in selected.centroid_distance]
    axes[1].barh(selected.naics4, selected.centroid_distance, color=colors)
    axes[1].set(title="Historical classification changes", xlabel="Distance to 3-digit centroid")
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def embedding_display_comparison(vectors, labels, destination):
    coords = [umap.UMAP(random_state=SEED).fit_transform(vectors),
              TSNE(n_components=2, random_state=SEED, init="pca",
                   perplexity=min(30, len(vectors) - 1)).fit_transform(vectors)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, xy, name in zip(axes, coords, ["UMAP", "t-SNE"]):
        ax.scatter(xy[:, 0], xy[:, 1], c=labels.loc[vectors.index], cmap="tab10", s=20)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def taxonomy_agreement(labels, taxonomy):
    common = labels.index.intersection(taxonomy.index)
    return {"nmi": normalized_mutual_info_score(labels.loc[common], taxonomy.loc[common]),
            "ari": adjusted_rand_score(labels.loc[common], taxonomy.loc[common])}


def representative_industries(graph, labels, top_n=10):
    """Rank industries by weighted connections within their own cluster."""
    rows = []
    for cluster in sorted(labels.unique()):
        members = set(labels[labels.eq(cluster)].index)
        for node in members:
            within_strength = sum(graph[node][neighbor]["weight"]
                                  for neighbor in graph.neighbors(node) if neighbor in members)
            rows.append((cluster, node, within_strength))
    return pd.DataFrame(rows, columns=["cluster", "naics4", "within_strength"]).sort_values(
        ["cluster", "within_strength"], ascending=[True, False]).groupby("cluster").head(top_n)



def lead_space_comparison(full_phi, full_graph, full_labels, lead_phi, lead_graph, lead_labels):
    """Compare full and lead-investor industry spaces on common industries."""
    nodes = sorted(set(full_graph.nodes) & set(lead_graph.nodes))
    pairs = list(itertools.combinations(nodes, 2))
    full = [full_phi.at[i, j] for i, j in pairs]
    lead = [lead_phi.at[i, j] for i, j in pairs]
    full_edges = {tuple(sorted(edge)) for edge in full_graph.subgraph(nodes).edges}
    lead_edges = {tuple(sorted(edge)) for edge in lead_graph.subgraph(nodes).edges}
    return {"matched_nodes": len(nodes),
            "proximity_spearman": spearmanr(full, lead).statistic,
            "edge_jaccard": len(full_edges & lead_edges) / len(full_edges | lead_edges),
            "cluster_nmi": normalized_mutual_info_score(full_labels.loc[nodes], lead_labels.loc[nodes]),
            "cluster_ari": adjusted_rand_score(full_labels.loc[nodes], lead_labels.loc[nodes])}




def main():
    """Evaluate cluster structure and write Figure 4 and SI displays."""
    from ..settings import OUT_STRUCTURE, OUT_SI
    from .space import load_space
    OUT_STRUCTURE.mkdir(parents=True, exist_ok=True)
    OUT_SI.mkdir(parents=True, exist_ok=True)
    _, phi, graph, vectors, labels = load_space("full")
    pairs = within_cluster_cosines(vectors, labels)
    ranking = rank_historical_changes(vectors, "China")
    plot_internal_similarity(pairs, ranking, "China", OUT_STRUCTURE / "structure.png")
    embedding_display_comparison(vectors, labels, OUT_SI / "embedding_comparison.png")
    taxonomy = pd.read_csv(SUPPLEMENT / "taxonomy_groups.csv", dtype={"naics4": str})
    taxonomy = taxonomy.loc[taxonomy.country.eq("China")].set_index("naics4").taxonomy_group
    print("Within/between cluster cosine:")
    print(pairs.groupby("same_cluster").cosine.agg(["count", "mean"]).to_string())
    print("Historical classification changes:")
    print(ranking.to_string(index=False))
    print("Representative industries:")
    print(representative_industries(graph, labels).to_string(index=False))
    print("Taxonomy agreement:", taxonomy_agreement(labels, taxonomy))
    _, lead_phi, lead_graph, _, lead_labels = load_space("lead")
    print("Lead/full comparison:", lead_space_comparison(
        phi, graph, labels, lead_phi, lead_graph, lead_labels))


if __name__ == "__main__":
    main()
