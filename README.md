# VC-Revealed Industry Relatedness

Research code for studying industry relatedness revealed by venture-capital (VC) investment and its association with industrial diversification in China and the United States.

## Repository structure

```text
china/script/       China analyses
us/script/          United States analyses
requirements.txt    Python dependencies
```

Each country's `script/` package contains:

- `preparation/`: screen investment records and harmonize industry classifications.
- `industry_space/`: construct the industry-relatedness network, compute embeddings and clusters, and evaluate structural features.
- `external_validation/`: compare VC-based relatedness with trade-based relatedness.
- `null_model/`: perform null-model comparisons.
- `kbv/`: prepare activation panels, estimate statistical models, and run instrumental-variable diagnostics.
- `settings.py`: paths and analysis parameters.

## Installation

From the repository root, create a Python environment and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The package versions are not pinned in this archive. Results may depend on the runtime environment and library versions.

## Data availability and reproduction

**The source data are NOT included in this repository.** The scripts read country-specific input datasets from `china/database/` and `us/database/` (see `settings.py` and the relevant scripts). Some inputs may originate from third-party commercial databases and cannot be redistributed without permission. You must obtain and prepare any required data separately, consistent with their licenses and data-use agreements.

Examples of expected inputs include:

- China: `investments.xlsx`, `investors.xlsx`, `exits.xlsx`, `listings.xlsx`, `acquisitions.xlsx`, and classification/mapping CSV files.
- United States: `funding_rounds.csv`, `investors.csv`, `companies.csv`, `acquisitions.csv`, and classification/mapping CSV files.
- Both: supplementary classification, validation, and model-control inputs, including `naics_structure.csv`, `trade_relatedness.csv`, `taxonomy_groups.csv`, and others referred to in the scripts.

Because the raw and supplementary datasets are absent, the code **cannot reproduce all published findings from this archive alone**. Outputs are written beneath each country's `result/` directory when the necessary inputs are supplied.

## Suggested execution order

Run commands from the repository root. Do each stage for a country after placing its required inputs in that country's `database/` directory:

```bash
python -m china.script.preparation.records
python -m china.script.preparation.harmonize
python -m china.script.industry_space.space
python -m china.script.industry_space.structure
python -m china.script.external_validation.trade
python -m china.script.null_model.run
python -m china.script.kbv.panel
python -m china.script.kbv.models
python -m china.script.kbv.iv.estimate
```

For U.S. analyses, replace `china` with `us` in the commands. Some validation and modeling steps require additional input files; consult individual scripts for the expected filenames and formats.

**Note:** Execution order is based on apparent input/output dependencies in the scripts; these commands have not been end-to-end validated without the underlying research data.

## Citation

If you use this software, please cite the associated research article. Add its final bibliographic details and DOI after publication.

## License

Add a `LICENSE` file before describing the repository as fully open-source. Choose a license only after confirming that all code owners agree and that none of the code is subject to third-party restrictions.
