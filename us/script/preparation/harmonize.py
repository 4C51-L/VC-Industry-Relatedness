from __future__ import annotations
import pandas as pd
from ..settings import SUPPLEMENT, RAW

def industry_participations(events, country):
    crosswalk = pd.read_csv(SUPPLEMENT / "manual_industry_map.csv", dtype={"naics4": str})
    crosswalk = crosswalk.loc[(crosswalk.country == country) &
                              crosswalk.review_status.eq("approved")]
    parts = events.assign(source_label=events.industry_label.fillna("").str.split(r"[,;|]"))
    parts = parts.explode("source_label")
    parts["source_label"] = parts.source_label.str.strip()
    parts = parts.merge(crosswalk[["source_label", "naics4"]], on="source_label", how="inner")
    parts["naics4"] = parts.naics4.str.zfill(4)
    parts = parts.loc[parts.naics4.str.fullmatch(r"\d{4}")]
    official = pd.read_csv(RAW / "naics_structure.csv", dtype={"code": str})
    valid = set(official.loc[official.code.str.fullmatch(r"\d{4}"), "code"])
    parts = parts.loc[parts.naics4.isin(valid)]
    return parts.drop_duplicates(["country", "event_id", "investor_id", "naics4"])

def success_exits_us(company, acquisitions):
    c = company[["company_id", "ipo_date"]].copy()
    c["exit_date"] = pd.to_datetime(c.ipo_date, errors="coerce")
    c = c.dropna(subset=["exit_date"])[["company_id", "exit_date"]]
    a = acquisitions[["company_id", "acquisition_id", "acquisition_date"]].copy()
    flag = pd.read_csv(SUPPLEMENT / "high_value_ma_flag.csv")
    a = a.merge(flag[["country", "acquisition_id", "is_high_value"]].query("country == 'US'"),
                on="acquisition_id", how="inner")
    a = a.loc[a.is_high_value.eq(1)]
    a["exit_date"] = pd.to_datetime(a.acquisition_date, errors="coerce")
    return pd.concat([c, a[["company_id", "exit_date"]]]).dropna().drop_duplicates()


def main():
    """Map screened investments and qualify successful exits."""
    import json
    from ..settings import OUT_PREP
    OUT_PREP.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(OUT_PREP / "screened_participations.csv", low_memory=False)
    mapped = industry_participations(events, "US")
    mapped.to_csv(OUT_PREP / "industry_participations.csv", index=False)

    company = pd.read_csv(RAW / "companies.csv", low_memory=False).rename(columns={
        "Organization Name URL": "company_id", "IPO Date": "ipo_date"})
    acquisitions = pd.read_csv(RAW / "acquisitions.csv", low_memory=False).rename(columns={
        "Transaction Name URL": "acquisition_id", "Acquiree Name URL": "company_id",
        "Announced Date": "acquisition_date"})
    exits = success_exits_us(company, acquisitions)
    exits.to_csv(OUT_PREP / "qualified_exits.csv", index=False)
    audit = json.loads((OUT_PREP / "screening_audit.json").read_text(encoding="utf-8"))
    coverage = pd.Series({"country": "US", **audit,
        "mapped_participations": len(mapped),
        "investors": mapped.investor_id.nunique(),
        "companies": mapped.company_id.nunique(),
        "industries": mapped.naics4.nunique()}, name="sample_coverage")
    print(coverage.to_string())


if __name__ == "__main__":
    main()
