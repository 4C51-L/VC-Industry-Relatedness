from __future__ import annotations
import numpy as np
import pandas as pd
from ..settings import PERIODS
from ..industry_space.space import rca, relatedness


def period_of(year):
    for index, (start, end) in enumerate(PERIODS):
        if start <= year <= end:
            return index
    return np.nan


def period_matrices(participations):
    universe = sorted(participations.naics4.unique())
    investors = sorted(participations.investor_id.unique())
    matrices = {}
    proximities = {}
    for period, (start, end) in enumerate(PERIODS):
        subset = participations.loc[participations.year.between(start, end)]
        matrix = rca(subset, universe).reindex(index=investors, columns=universe, fill_value=0)
        matrices[period] = matrix
        proximities[period] = relatedness(matrix)
    return matrices, proximities


def density_matrix(rca_matrix, phi):
    active = (rca_matrix > 1).astype(float)
    weights = phi.reindex(index=active.columns, columns=active.columns).fillna(0)
    numerator = active.to_numpy() @ weights.to_numpy()
    denominator = weights.sum(axis=0).replace(0, np.nan).to_numpy()
    return pd.DataFrame(numerator / denominator, index=active.index,
                        columns=active.columns).fillna(0)


def _event_success_counts(participations, exits, prior_period, cumulative):
    cutoff = PERIODS[prior_period][1]
    start = PERIODS[0][0] if cumulative else PERIODS[prior_period][0]
    cohorts = participations.loc[participations.year.between(start, cutoff),
                                 ["investor_id", "event_id", "company_id", "event_date"]].drop_duplicates()
    observed = exits.loc[exits.exit_date.dt.year.le(cutoff), ["company_id", "exit_date"]]
    successful = cohorts.merge(observed, on="company_id", how="inner")
    successful = successful.loc[successful.exit_date.ge(pd.to_datetime(successful.event_date,
                                                                      errors="coerce"))]
    return successful.drop_duplicates(["investor_id", "event_id"]).groupby("investor_id").size()


def _lagged_controls(participations, prior_period):
    cutoff = PERIODS[prior_period][1]
    history = participations.loc[participations.year.le(cutoff)]
    breadth = history.groupby("investor_id").naics4.nunique().rename("breadth")
    experience = history.drop_duplicates(["investor_id", "event_id"]).groupby("investor_id").size().rename("experience")
    unique = history.drop_duplicates(["investor_id", "event_id"]).copy()
    if "syndicated" not in unique:
        round_size = history.groupby("event_id").investor_id.nunique()
        unique["syndicated"] = unique.event_id.map(round_size).gt(1).astype(int)
    syndication = unique.groupby("investor_id").syndicated.mean().rename("syndication_share")
    return pd.concat([breadth, experience, syndication], axis=1)


def build_transition_panel(participations, exits, lead_only=False):
    participations = participations.copy()
    round_size = participations.groupby("event_id").investor_id.nunique()
    participations["syndicated"] = participations.event_id.map(round_size).gt(1).astype(int)
    if lead_only:
        if participations.is_lead.isna().any():
            raise ValueError("Lead status must be supplied for every lead-only source record")
        participations = participations.loc[participations.is_lead.eq(True)].copy()
    matrices, proximities = period_matrices(participations)
    region = participations.sort_values("year").drop_duplicates("investor_id").set_index("investor_id").region
    rows = []
    for period in range(1, len(PERIODS) - 1):
        prior = period - 1
        current = matrices[period]
        future = matrices[period + 1]
        density = density_matrix(current, proximities[period])
        controls = _lagged_controls(participations, prior)
        adjacent = _event_success_counts(participations, exits, prior, cumulative=False)
        cumulative = _event_success_counts(participations, exits, prior, cumulative=True)
        risk = (current <= 1).stack().rename("at_risk")
        frame = pd.DataFrame({"at_risk": risk,
            "activation": (future > 1).stack(),
            "exit": ((current > 1) & (future <= 1)).stack(),
            "density": density.stack(),
            "lagged_rca": matrices[prior].stack()}).reset_index()
        frame = frame.rename(columns={"level_0": "investor_id", "level_1": "naics4"})
        frame = frame.loc[frame.at_risk]
        frame["period"] = period
        frame["region"] = frame.investor_id.map(region)
        frame["success_adjacent"] = frame.investor_id.map(adjacent).fillna(0)
        frame["success_cumulative"] = frame.investor_id.map(cumulative).fillna(0)
        frame = frame.join(controls, on="investor_id")
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel["lead_only"] = lead_only
    return panel, matrices, proximities


def main():
    """Build full and lead-only activation panels and save period matrices."""
    from ..settings import OUT_PREP, OUT_KBV
    OUT_KBV.mkdir(parents=True, exist_ok=True)
    mapped = pd.read_csv(OUT_PREP / "industry_participations.csv",
                         dtype={"naics4": str}, low_memory=False)
    exits = pd.read_csv(OUT_PREP / "qualified_exits.csv", parse_dates=["exit_date"])
    for kind, lead_only in (("full", False), ("lead", True)):
        sample, matrices, proximities = build_transition_panel(mapped, exits, lead_only)
        sample.to_csv(OUT_KBV / f"{kind}_activation_panel.csv", index=False)
        if kind == "full":
            for period in matrices:
                matrices[period].to_csv(OUT_KBV / f"{kind}_period_{period}_rca.csv")
                proximities[period].to_csv(OUT_KBV / f"{kind}_period_{period}_relatedness.csv")
        print(f"{kind} activation panel: {len(sample)} observations")


if __name__ == "__main__":
    main()
