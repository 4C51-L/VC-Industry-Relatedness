from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from ..models import prepare_model_sample


def peer_rca_instrument(panel, matrices, proximities):
    out = []
    regions = panel[["investor_id", "region"]].drop_duplicates().set_index("investor_id").region
    for period in sorted(panel.period.unique()):
        prior = period - 1
        rca = matrices[prior]
        phi = proximities[prior].reindex(index=rca.columns, columns=rca.columns).fillna(0)
        location = regions.reindex(rca.index)
        peer_sum = rca.groupby(location).transform("sum")
        peer_n = location.map(location.value_counts())
        leave_one_out = (peer_sum - rca).div((peer_n - 1).replace(0, np.nan), axis=0)
        weighted = leave_one_out.to_numpy() @ phi.to_numpy()
        denominator = phi.sum(axis=0).replace(0, np.nan).to_numpy()
        instrument = pd.DataFrame(weighted / denominator, index=rca.index, columns=rca.columns)
        stacked = instrument.stack(dropna=False).rename("iv_raw").reset_index()
        stacked = stacked.rename(columns={"level_0": "investor_id", "level_1": "naics4"})
        stacked["period"] = period
        out.append(stacked)
    result = panel.merge(pd.concat(out), on=["investor_id", "naics4", "period"], how="left")
    return result


def fit_two_stage(data, performance="success_adjacent"):
    common = f"{performance}_z + lagged_rca_z + breadth_z + experience_z + syndication_share_z + C(investor_id) + C(period)"
    first = smf.ols("density_z ~ iv_z + " + common, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data.region})
    stage = data.copy()
    stage["first_residual"] = data.density_z - first.predict(data)
    second = smf.probit("activation ~ density_z + first_residual + " + common,
                        data=stage).fit(disp=False, maxiter=200,
                                        cov_type="cluster", cov_kwds={"groups": stage.region})
    restricted = smf.ols("density_z ~ " + common, data=data).fit()
    partial_r2 = (restricted.ssr - first.ssr) / restricted.ssr
    first_f = float(first.wald_test("iv_z = 0", use_f=True, scalar=True).statistic)
    return first, second, {"partial_r2": partial_r2, "first_stage_F": first_f}


def bootstrap_by_region(data, performance="success_adjacent", repetitions=999, seed=42):
    rng = np.random.default_rng(seed)
    regions = data.region.dropna().unique()
    estimates = []
    for draw in range(repetitions):
        sampled = rng.choice(regions, size=len(regions), replace=True)
        pieces = []
        for copy, region in enumerate(sampled):
            chunk = data.loc[data.region.eq(region)].copy()
            chunk["investor_id"] = chunk.investor_id.astype(str) + f"__{copy}"
            chunk["region"] = f"{region}__{copy}"
            pieces.append(chunk)
        trial = pd.concat(pieces, ignore_index=True)
        try:
            _, second, _ = fit_two_stage(trial, performance)
            estimates.append(second.params)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pd.DataFrame(estimates)


def estimate_iv(panel, matrices, proximities, performance="success_adjacent", bootstrap_repetitions=999):
    with_iv = peer_rca_instrument(panel, matrices, proximities)
    data = prepare_model_sample(with_iv.dropna(subset=["iv_raw"]))
    data["iv_z"] = (data.iv_raw - data.iv_raw.mean()) / data.iv_raw.std(ddof=0)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv_z"])
    first, second, diagnostics = fit_two_stage(data, performance)
    draws = bootstrap_by_region(data, performance, repetitions=bootstrap_repetitions)
    diagnostics["bootstrap_se"] = draws.std(ddof=1).to_dict()
    residual_se = draws.first_residual.std(ddof=1)
    residual_z = second.params["first_residual"] / residual_se
    diagnostics["residual_bootstrap_z"] = residual_z
    diagnostics["residual_bootstrap_p"] = 2 * norm.sf(abs(residual_z))
    return first, second, diagnostics


def main():
    """Print IV-Probit stages and geography-cluster bootstrap diagnostics."""
    import argparse
    from ...settings import OUT_KBV, PERIODS
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-repetitions", type=int, default=999)
    args = parser.parse_args()
    panel = pd.read_csv(OUT_KBV / "full_activation_panel.csv",
                        dtype={"naics4": str}, low_memory=False)
    matrices, proximities = {}, {}
    for period in range(len(PERIODS)):
        matrices[period] = pd.read_csv(OUT_KBV / f"full_period_{period}_rca.csv", index_col=0)
        proximities[period] = pd.read_csv(
            OUT_KBV / f"full_period_{period}_relatedness.csv", index_col=0)
        matrices[period].columns = matrices[period].columns.map(str)
        proximities[period].index = proximities[period].index.map(str)
        proximities[period].columns = proximities[period].columns.map(str)
    for performance in ("success_adjacent", "success_cumulative"):
        first, second, diagnostics = estimate_iv(
            panel, matrices, proximities, performance, args.bootstrap_repetitions)
        print(f"{performance} first stage:\n{first.summary2().tables[1].to_string()}")
        print(f"{performance} second stage:\n{second.summary2().tables[1].to_string()}")
        print(f"{performance} diagnostics:\n{pd.Series(diagnostics).to_string()}")


if __name__ == "__main__":
    main()
