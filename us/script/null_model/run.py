from __future__ import annotations
import random

import numpy as np
from scipy import sparse


def rows_from_matrix(matrix: sparse.spmatrix | np.ndarray) -> list[set[int]]:
    """Convert a binary investor-industry matrix to row sets."""
    csr = sparse.csr_matrix(matrix, dtype=np.int8)
    csr.sort_indices()
    return [set(map(int, csr.indices[csr.indptr[i]:csr.indptr[i + 1]]))
            for i in range(csr.shape[0])]


def matrix_from_rows(rows: list[set[int]], n_columns: int) -> sparse.csr_matrix:
    """Rebuild a binary matrix from randomized row sets."""
    indices = np.fromiter((j for row in rows for j in sorted(row)), dtype=np.int32)
    indptr = np.empty(len(rows) + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(np.fromiter((len(row) for row in rows), dtype=np.int32), out=indptr[1:])
    data = np.ones(len(indices), dtype=np.int32)
    return sparse.csr_matrix((data, indices, indptr), shape=(len(rows), n_columns))


def curveball(base_rows: list[set[int]], trades: int, seed: int) -> tuple[list[set[int]], int]:
    """Trade row-exclusive industries while preserving both matrix margins."""
    rows = [row.copy() for row in base_rows]
    rng = random.Random(seed)
    changed = 0
    n = len(rows)
    for _ in range(trades):
        a = rng.randrange(n)
        b = rng.randrange(n - 1)
        if b >= a:
            b += 1
        only_a = rows[a] - rows[b]
        only_b = rows[b] - rows[a]
        if not only_a or not only_b:
            continue
        shared = rows[a] & rows[b]
        pool = sorted(only_a | only_b)
        rng.shuffle(pool)
        cut = len(only_a)
        rows[a] = shared | set(pool[:cut])
        rows[b] = shared | set(pool[cut:])
        changed += 1
    return rows, changed


def validate_margins(base: sparse.csr_matrix, sampled: sparse.csr_matrix) -> None:
    """Check that the randomized matrix has the original row and column totals."""
    assert sampled.shape == base.shape
    assert sampled.nnz == base.nnz
    assert np.array_equal(np.asarray(sampled.sum(axis=1)), np.asarray(base.sum(axis=1)))
    assert np.array_equal(np.asarray(sampled.sum(axis=0)), np.asarray(base.sum(axis=0)))
    assert np.all(sampled.data == 1)


import argparse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from ..settings import OUT_NULL, SEED
from ..industry_space.space import weighted_network, disparity_backbone, embed_and_cluster, load_space
from ..industry_space.structure import within_cluster_cosines

def fixed_margin_randomize(binary, steps, rng):
    from scipy import sparse
    source = sparse.csr_matrix(binary.to_numpy().astype(int))
    rows, _ = curveball(rows_from_matrix(source), steps, int(rng.integers(0, 2**31)))
    sampled = matrix_from_rows(rows, binary.shape[1])
    validate_margins(source, sampled)
    return pd.DataFrame(sampled.toarray(), index=binary.index, columns=binary.columns)


def null_network_statistics(rca_matrix, observed_labels, repetitions=999):
    observed = (rca_matrix > 1).astype(int)
    rng = np.random.default_rng(SEED)
    records = []
    for draw in range(repetitions):
        randomized = fixed_margin_randomize(observed, 10 * len(observed), rng)
        overlap = randomized.T.dot(randomized)
        n = randomized.sum(axis=0).replace(0, np.nan)
        phi = np.minimum(overlap.div(n, axis=0), overlap.div(n, axis=1)).fillna(0)
        np.fill_diagonal(phi.values, 0)
        graph = disparity_backbone(weighted_network(phi))
        graph = graph.subgraph(observed_labels.index).copy()
        if graph.number_of_edges() == 0:
            continue
        groups = [set(observed_labels[observed_labels.eq(k)].index) & set(graph.nodes)
                  for k in observed_labels.unique()]
        groups = [group for group in groups if group]
        records.append((draw, nx.community.modularity(graph, groups, weight="weight")))
    return pd.DataFrame(records, columns=["draw", "modularity"])


def embedding_separation_null(rca_matrix, observed_difference, repetitions=1000):
    binary = (rca_matrix > 1).astype(int)
    rng = np.random.default_rng(SEED)
    values = []
    for draw in range(repetitions):
        shuffled = fixed_margin_randomize(binary, 10 * len(binary), rng)
        overlap = shuffled.T.dot(shuffled)
        n = shuffled.sum(axis=0).replace(0, np.nan)
        phi = np.minimum(overlap.div(n, axis=0), overlap.div(n, axis=1)).fillna(0)
        np.fill_diagonal(phi.values, 0)
        graph = disparity_backbone(weighted_network(phi))
        if graph.number_of_nodes() < 5:
            continue
        vectors, labels = embed_and_cluster(graph)
        pairs = within_cluster_cosines(vectors, labels)
        values.append((draw, pairs.loc[pairs.same_cluster, "cosine"].mean() -
                       pairs.loc[~pairs.same_cluster, "cosine"].mean()))
    result = pd.DataFrame(values, columns=["draw", "difference"])
    result.attrs["empirical_p"] = (1 + (result.difference >= observed_difference).sum()) / (1 + len(result))
    return result


def plot_null(observed_value, null_values, destination, label):
    plt.figure(figsize=(7, 4))
    plt.hist(null_values, bins=35, color="lightsteelblue")
    plt.axvline(observed_value, color="crimson", linewidth=2)
    plt.xlabel(label)
    plt.tight_layout()
    plt.savefig(destination, dpi=300)
    plt.close()




def main():
    """Run fixed-margin network and embedding null comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-repetitions", type=int, default=999)
    parser.add_argument("--embedding-repetitions", type=int, default=1000)
    args = parser.parse_args()
    OUT_NULL.mkdir(parents=True, exist_ok=True)
    rca_matrix, _, graph, vectors, labels = load_space("full")
    pairs = within_cluster_cosines(vectors, labels)
    groups = [set(labels[labels.eq(k)].index) for k in sorted(labels.unique())]
    observed_q = nx.community.modularity(graph, groups, weight="weight")
    simulated_q = null_network_statistics(rca_matrix, labels, args.network_repetitions)
    plot_null(observed_q, simulated_q.modularity, OUT_NULL / "modularity_null.png", "Modularity")
    observed_gap = pairs.loc[pairs.same_cluster, "cosine"].mean() - pairs.loc[~pairs.same_cluster, "cosine"].mean()
    simulated_gap = embedding_separation_null(rca_matrix, observed_gap, args.embedding_repetitions)
    plot_null(observed_gap, simulated_gap.difference, OUT_NULL / "embedding_null.png", "Within minus between cosine")
    print("Observed modularity:", observed_q)
    print("Null modularity:", simulated_q.modularity.describe().to_string())
    print("Observed embedding gap:", observed_gap)
    print("Null embedding gap:", simulated_gap.difference.describe().to_string())
    print("Embedding empirical p:", simulated_gap.attrs["empirical_p"])


if __name__ == "__main__":
    main()
