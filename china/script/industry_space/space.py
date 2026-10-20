from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from node2vec import Node2Vec
import umap
import matplotlib.pyplot as plt

from ..settings import ALPHA, EMBEDDING_DIM, WALK_LENGTH, WALKS_PER_NODE, WINDOW, EPOCHS, N_CLUSTERS, SEED


ALPHA = 0.1

def calc_rca(matrix, investors):
    """RCA：VC 在某行业投资占比 ÷ 该行业投资占全部投资比例（论文式 1）。"""
    row_sum = matrix.sum(axis=1)
    keep = row_sum > 0
    if not keep.all():
        print("警告：%d 家 VC 没有任何有码投资记录，已剔除。" % int((~keep).sum()))
        matrix = matrix[keep]
        investors = [v for v, k in zip(investors, keep) if k]
        row_sum = row_sum[keep]
    total = float(matrix.sum())
    col_sum = matrix.sum(axis=0)
    rca = (matrix / row_sum[:, None]) / (col_sum[None, :] / total)
    return rca, investors

def calc_proximity(rca):
    """产业邻近性 φ[j,k] = min(P(j|k), P(k|j))（论文式 2）；对称、对角为 0。"""
    b = (rca > 1).astype(np.float64)
    common = b.T @ b                    
    alone = np.diag(common).copy()      
    n_zero = int((alone == 0).sum())
    if n_zero:
        print("警告：%d 个行业没有 VC 达到 RCA>1，其邻近性按 0 处理。" % n_zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_jk = common / alone[None, :]
        p_kj = common / alone[:, None]
    phi = np.minimum(p_jk, p_kj)
    phi = np.nan_to_num(phi, nan=0.0)
    np.fill_diagonal(phi, 0.0)
    return phi

def disparity_filter(graph, weight="weight"):
    """计算无向边两端显著性；alpha 为两端较小值，端点值均四位小数舍入。"""
    endpoint_alpha = {}
    edge_weight = {}
    for u in graph:
        k = len(graph[u])                       
        if k == 1:
            v = next(iter(graph[u]))
            key = tuple(sorted((u, v)))
            endpoint_alpha.setdefault(key, {})[u] = 0.0
            edge_weight[key] = graph[u][v][weight]
        elif k > 1:
            sum_w = sum(np.absolute(graph[u][v][weight]) for v in graph[u])
            for v in graph[u]:
                w = graph[u][v][weight]
                p_ij = float(np.absolute(w)) / sum_w        
                alpha_ij = (1.0 - p_ij) ** (k - 1)   
                key = tuple(sorted((u, v)))
                endpoint_alpha.setdefault(key, {})[u] = float("%.4f" % alpha_ij)
                edge_weight[key] = w
    scored = nx.Graph()
    for (u, v), values in endpoint_alpha.items():
        alpha_u, alpha_v = values.get(u, 1.0), values.get(v, 1.0)
        scored.add_edge(u, v, weight=edge_weight[(u, v)], alpha_u=alpha_u,
                        alpha_v=alpha_v, alpha=min(alpha_u, alpha_v))
    return scored

def extract_backbone(graph, alpha=ALPHA):
    """按无向 OR 规则筛边：任一端 alpha < 阈值；之后删除孤立点。"""
    scored = disparity_filter(graph)
    backbone = nx.Graph([(u, v, d) for u, v, d in scored.edges(data=True)
                         if d["alpha_u"] < alpha or d["alpha_v"] < alpha])
    backbone.remove_nodes_from(list(nx.isolates(backbone)))
    return backbone


def rca(participations, industry_universe=None):
    if participations.empty:
        return pd.DataFrame(columns=industry_universe or [], dtype=float)
    counts = participations.groupby(["investor_id", "naics4"]).size().unstack(fill_value=0)
    if industry_universe is not None:
        counts = counts.reindex(columns=industry_universe, fill_value=0)
    active_columns = counts.columns[counts.sum(axis=0).gt(0)]
    values, investors = calc_rca(counts[active_columns].to_numpy(), list(counts.index))
    return pd.DataFrame(values, index=investors, columns=active_columns).reindex(
        columns=counts.columns, fill_value=0)


def relatedness(rca_matrix):
    return pd.DataFrame(calc_proximity(rca_matrix.to_numpy()),
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
    return extract_backbone(graph, alpha)


def embed_and_cluster(graph):
    walker = Node2Vec(graph, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH,
        num_walks=WALKS_PER_NODE, p=1, q=1, weight_key="weight", seed=SEED, workers=1)
    model = walker.fit(window=WINDOW, min_count=1, epochs=EPOCHS, seed=SEED)
    nodes = list(graph.nodes)
    vectors = pd.DataFrame([model.wv[str(node)] for node in nodes], index=nodes)
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
               "China" + (" lead" if kind == "lead" else ""))
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
