# VC-Revealed Industry Relatedness

Research code for studying industry relatedness revealed by venture-capital (VC) investment and its association with industrial diversification in China and the United States.

## Repository structure

```text
china/script/            China analyses
us/script/               United States analyses
requirements.txt         Python dependencies
```

Each country's `script/` package contains:

- `industry_space/`: construct VC-revealed industry-relatedness networks and compute industry embeddings, clusters, and network outputs.
- `external_validation/`: compare VC-relatedness with trade-based relatedness.
- `null_model/`: perform null-model comparisons.
- `kbv/`: construct industry-activation panels, estimate statistical models, and run instrumental-variable diagnostics.
- `settings.py`: input/output paths and analysis parameters.

## Data sources and preprocessing

Investment data cover 1990–2020 and originate from:

- **China:** Zero2IPO (PEdata) — https://max.pedata.cn/
- **United States:** Crunchbase — https://www.crunchbase.com/

**The source investment data were cleaned and harmonized before the analyses in this repository.** Preprocessing included screening investment records, applying sample-selection criteria, and harmonizing industry classifications to four-digit NAICS 2017 codes.

**The unified operations of data cleaning and classification have been completed in advance.** 
## Required analysis inputs

The scripts use country-specific paths set in `china/script/settings.py` and `us/script/settings.py`. For each country, the main analysis requires the following **preprocessed inputs**, prepared separately:

```text
china/result/preparation/industry_participations.csv
china/result/preparation/qualified_exits.csv
us/result/preparation/industry_participations.csv
us/result/preparation/qualified_exits.csv
```

Other analysis and validation steps may require additional inputs in the country-specific `database/` directory, such as `trade_relatedness.csv`, `taxonomy_groups.csv`, `naics_revision_relations.csv`, and `performance_tier_cutoffs.csv`; see the individual scripts for their schemas and usage. None of these datasets are included in this archive.

## Installation

From the repository root, create a Python environment and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependency versions are not pinned in this archive, and some results may depend on the runtime environment and package versions.

## Suggested execution order

**After obtaining or preparing the required preprocessed inputs**, run the analysis stages from the repository root:

```bash
python -m china.script.industry_space.space
python -m china.script.industry_space.structure
python -m china.script.external_validation.trade
python -m china.script.null_model.run
python -m china.script.kbv.panel
python -m china.script.kbv.models
python -m china.script.kbv.iv.estimate
```

For the U.S. analyses, replace `china` with `us`. Later stages depend on outputs from earlier stages and may require additional validation or model-control inputs. This execution sequence has not been end-to-end validated without the underlying research data.

## Citation

If you use this software, please cite the accompanying research article. Final publication details and DOI will be added when available.
