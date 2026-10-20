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

def success_exits_china(exits, listings, acquisitions):
    ipo = listings.rename(columns={"企业名称": "company_id", "上市时间": "exit_date"})
    ipo = ipo[["company_id", "exit_date"]]
    ma = acquisitions.rename(columns={"被并购方": "company_id", "完成时间": "exit_date"})
    ma["acquisition_id"] = ma.index.astype(str)
    flag = pd.read_csv(SUPPLEMENT / "high_value_ma_flag.csv", dtype={"acquisition_id": str})
    ma = ma.merge(flag.query("country == 'China'"), on="acquisition_id", how="inner")
    ma = ma.loc[ma.is_high_value.eq(1), ["company_id", "exit_date"]]
    direct = exits.rename(columns={"被退出方": "company_id", "退出时间": "exit_date",
                                   "退出方式": "exit_type"})
    direct = direct.loc[direct.exit_type.astype(str).str.contains("IPO|上市", case=False),
                        ["company_id", "exit_date"]]
    result = pd.concat([ipo, ma, direct]).dropna().drop_duplicates()
    result["exit_date"] = pd.to_datetime(result.exit_date, errors="coerce")
    return result.dropna(subset=["exit_date"])


def main():
    """Map screened investments and qualify successful exits."""
    import json
    from ..settings import OUT_PREP
    OUT_PREP.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(OUT_PREP / "screened_participations.csv", low_memory=False)
    mapped = industry_participations(events, "China")
    mapped.to_csv(OUT_PREP / "industry_participations.csv", index=False)

    raw_exits = pd.read_excel(RAW / "exits.xlsx")
    listings = pd.read_excel(RAW / "listings.xlsx")
    acquisitions = pd.read_excel(RAW / "acquisitions.xlsx")
    exits = success_exits_china(raw_exits, listings, acquisitions)
    exits.to_csv(OUT_PREP / "qualified_exits.csv", index=False)
    audit = json.loads((OUT_PREP / "screening_audit.json").read_text(encoding="utf-8"))
    coverage = pd.Series({"country": "China", **audit,
        "mapped_participations": len(mapped),
        "investors": mapped.investor_id.nunique(),
        "companies": mapped.company_id.nunique(),
        "industries": mapped.naics4.nunique()}, name="sample_coverage")
    print(coverage.to_string())


if __name__ == "__main__":
    main()
