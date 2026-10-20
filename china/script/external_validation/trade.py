from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from ..settings import SUPPLEMENT, SEED

def mantel_spearman(vc, trade, permutations=9999):
    common = sorted(set(vc.index) & set(trade.index))
    a = vc.loc[common, common].to_numpy()
    b = trade.loc[common, common].to_numpy()
    triangle = np.triu_indices(len(common), 1)
    observed = spearmanr(a[triangle], b[triangle]).statistic
    rng = np.random.default_rng(SEED)
    extremes = 0
    for _ in range(permutations):
        p = rng.permutation(len(common))
        shuffled = b[np.ix_(p, p)]
        extremes += abs(spearmanr(a[triangle], shuffled[triangle]).statistic) >= abs(observed)
    return observed, (extremes + 1) / (permutations + 1)

def validate_trade(phi, country, destination, permutations=9999):
    table = pd.read_csv(SUPPLEMENT / "trade_relatedness.csv", dtype={"naics_i": str, "naics_j": str})
    table = table.loc[table.country.eq(country)]
    trade = table.pivot(index="naics_i", columns="naics_j", values="relatedness")
    trade = trade.combine_first(trade.T)
    common = sorted(set(phi.index) & set(trade.index))
    rho, p = mantel_spearman(phi.loc[common, common], trade.loc[common, common], permutations)
    tri = np.triu_indices(len(common), 1)
    plt.figure(figsize=(6, 5))
    plt.scatter(phi.loc[common, common].to_numpy()[tri],
                trade.loc[common, common].to_numpy()[tri], s=7, alpha=.3)
    plt.xlabel("VC relatedness")
    plt.ylabel("Trade relatedness")
    plt.title(f"{country}: Spearman={rho:.3f}; Mantel p={p:.3g}")
    plt.tight_layout()
    plt.savefig(destination, dpi=300)
    plt.close()
    return {"country": country, "spearman": rho, "mantel_p": p, "pairs": len(tri[0])}


def main():
    """Compare VC and trade relatedness and draw the external-validation figure."""
    import argparse
    from ..settings import OUT_TRADE
    from ..industry_space.space import load_space
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=9999)
    args = parser.parse_args()
    OUT_TRADE.mkdir(parents=True, exist_ok=True)
    _, phi, _, _, _ = load_space("full")
    result = validate_trade(phi, "China", OUT_TRADE / "trade_validation.png", args.permutations)
    print(pd.Series(result, name="trade_validation").to_string())


if __name__ == "__main__":
    main()
