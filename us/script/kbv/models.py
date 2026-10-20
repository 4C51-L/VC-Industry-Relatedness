from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

LOG_COLUMNS = ("lagged_rca", "breadth", "experience",
               "success_adjacent", "success_cumulative")
STANDARD_COLUMNS = ("density", "syndication_share")


def prepare_model_sample(panel):
    data = panel.dropna(subset=["activation", "density", "lagged_rca", "breadth",
                                "experience", "syndication_share", "region"]).copy()
    data["activation"] = data.activation.astype(int)
    for column in LOG_COLUMNS:
        data[f"{column}_log"] = np.log1p(data[column].clip(lower=0))
        source = data[f"{column}_log"]
        data[f"{column}_z"] = (source - source.mean()) / source.std(ddof=0)
    for column in STANDARD_COLUMNS:
        source = data[column]
        data[f"{column}_z"] = (source - source.mean()) / source.std(ddof=0)
    needed = ["activation", "density_z", "lagged_rca_z", "breadth_z",
              "experience_z", "syndication_share_z", "success_adjacent_z",
              "success_cumulative_z", "investor_id", "period", "region"]
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)


def formula(performance, interaction):
    terms = ["density_z", f"{performance}_z", "lagged_rca_z", "breadth_z",
             "experience_z", "syndication_share_z", "C(investor_id)", "C(period)"]
    if interaction:
        terms.append(f"density_z:{performance}_z")
    return "activation ~ " + " + ".join(terms)


def fit_probit(panel, performance="success_adjacent", interaction=False):
    data = prepare_model_sample(panel)
    model = smf.probit(formula(performance, interaction), data=data).fit(disp=False, maxiter=200)
    return model, data


def table_specifications(panel):
    outputs = {}
    for measure in ("success_adjacent", "success_cumulative"):
        for interaction in (False, True):
            outputs[(measure, interaction)] = fit_probit(panel, measure, interaction)
    return outputs


def plot_predicted_curves(model, data, performance, tiers, destination):
    grid = np.linspace(data.density_z.quantile(.01), data.density_z.quantile(.99), 100)
    fig, ax = plt.subplots(figsize=(7, 5))
    for tier, value in tiers.items():
        values = []
        for density in grid:
            scenario = data.copy()
            scenario["density_z"] = density
            scenario[f"{performance}_z"] = value
            values.append(model.predict(scenario).mean())
        ax.plot(grid, values, label=tier)
    ax.set(xlabel="Relatedness density (standardized)",
           ylabel="Predicted activation probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def load_performance_tiers(path, country, performance):
    rows = pd.read_csv(path)
    rows = rows.loc[rows.country.eq(country) & rows.performance.eq(performance)]
    return dict(zip(rows.tier, rows.performance_z))


def descriptive_and_correlation(model_sample):
    """Return descriptive statistics and correlations for the model sample."""
    variables = ["activation", "density", "lagged_rca_log", "breadth_log",
                 "experience_log", "syndication_share", "success_adjacent_log",
                 "success_cumulative_log"]
    data = model_sample[variables]
    return data.describe().T, data.corr(method="pearson")


def main():
    """Print KBV model tables and draw predicted-activation curves."""
    from ..settings import OUT_KBV, SUPPLEMENT
    OUT_KBV.mkdir(parents=True, exist_ok=True)
    for kind in ("full", "lead"):
        panel = pd.read_csv(OUT_KBV / f"{kind}_activation_panel.csv", low_memory=False)
        sample = prepare_model_sample(panel)
        descriptive, correlation = descriptive_and_correlation(sample)
        print(f"{kind} descriptive statistics:\n{descriptive.to_string()}")
        print(f"{kind} correlation matrix:\n{correlation.to_string()}")
        for (performance, interaction), (model, data) in table_specifications(panel).items():
            print(f"{kind} {performance} interaction={interaction} Probit:\n"
                  + model.summary2().tables[1].to_string())
            if kind == "full" and interaction:
                tiers = load_performance_tiers(SUPPLEMENT / "performance_tier_cutoffs.csv",
                                               "US", performance)
                plot_predicted_curves(model, data, performance, tiers,
                                      OUT_KBV / f"{performance}_predictions.png")


if __name__ == "__main__":
    main()
