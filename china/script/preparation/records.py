from __future__ import annotations
import re
import pandas as pd
from ..settings import RAW, SUPPLEMENT


def normalize_header(h):
    """表头归一化：去空白、全角括号转半角，便于列名匹配。"""
    if h is None:
        return ""
    return str(h).strip().replace("（", "(").replace("）", ")")


def _year(value):
    return pd.to_datetime(value, errors="coerce").dt.year

def _investor_kind(value):
    labels = {x.strip().lower() for x in re.split(r"[,;|]", str(value))}
    return bool(labels & {"micro vc", "venture capital", "corporate venture capital"})

def _window(df):
    return df.loc[df.year.between(1990, 2020)].copy()

def _exclude_rounds(df):
    return df.loc[~df.round_type.fillna("").str.contains(
        r"angel|天使|post[- ]?ipo|上市定增|上市后|ipo后|^--$", case=False,
        regex=True)].copy()

def _qualify(df):
    df = df.drop_duplicates(["country", "event_id", "investor_id"])
    n = df.groupby("investor_id").event_id.transform("size")
    return df.loc[n >= 10].copy()

def _normalize_regions(rows, country):
    mapping = pd.read_csv(SUPPLEMENT / "region_normalization.csv")
    mapping = mapping.loc[mapping.country.eq(country), ["source_region", "region"]]
    rows = rows.rename(columns={"region": "source_region"})
    return rows.merge(mapping, on="source_region", how="inner")

def read_records():
    events = pd.read_excel(RAW / "investments.xlsx", sheet_name="总表")
    events.columns = [normalize_header(col) for col in events.columns]
    raw_events = len(events)
    active = pd.read_excel(RAW / "investors.xlsx", sheet_name="活跃机构")
    all_investors = pd.read_excel(RAW / "investors.xlsx", sheet_name="全部机构")
    kind_map = pd.read_csv(SUPPLEMENT / "china_investor_type_map.csv")
    lead = pd.read_csv(SUPPLEMENT / "china_lead_status.csv")
    events = events.rename(columns={"投资方": "investor_name", "所属品牌机构": "brand",
        "项目": "project", "投资时间": "event_date", "投资轮次": "round_type",
        "行业": "industry_label", "企业名称": "company_name", "投资方所在地": "region"})
    events["year"] = _year(events.event_date)
    events = _exclude_rounds(_window(events))
    events["event_id"] = events[["company_name", "event_date", "round_type", "project"]].astype(str).agg("|".join, axis=1)
    events["investor_id"] = events.brand.fillna(events.investor_name).astype(str).str.strip()
    events["company_id"] = events.company_name.astype(str).str.strip()
    types = active[["VC公司名称", "机构类型", "总部"]].rename(columns={
        "VC公司名称": "investor_id", "机构类型": "raw_type", "总部": "headquarters"})
    backup = all_investors[["VC公司名称", "公司类型", "总部所在城市"]].rename(columns={
        "VC公司名称": "investor_id", "公司类型": "raw_type", "总部所在城市": "headquarters"})
    types = pd.concat([types, backup], ignore_index=True).drop_duplicates("investor_id")
    types = types.merge(kind_map, on="raw_type", how="left")
    types = types.loc[types.eligible_type.eq(1)]
    events = events.merge(types[["investor_id", "headquarters"]], on="investor_id", how="inner")
    events["region"] = events.headquarters
    events["country"] = "China"
    events = _normalize_regions(events, "China")
    events = events.merge(lead[["event_id", "investor_id", "is_lead"]],
                          on=["event_id", "investor_id"], how="left")
    if events.is_lead.isna().any():
        raise ValueError("china_lead_status.csv lacks lead status for some investment participations")
    events = events.dropna(subset=["event_id", "company_id", "investor_id", "industry_label"])
    for field in ("event_id", "company_id", "investor_id", "industry_label"):
        events[field] = events[field].astype("string").str.strip()
        events = events.loc[events[field].ne("") & events[field].ne("nan")]
    screened = len(events)
    events = _qualify(events)
    events.attrs["audit"] = {"raw_source_rows": raw_events,
                             "screened_participations": screened,
                             "qualified_participations": len(events)}
    exits = pd.read_excel(RAW / "exits.xlsx")
    listings = pd.read_excel(RAW / "listings.xlsx")
    acquisitions = pd.read_excel(RAW / "acquisitions.xlsx")
    return events, exits, listings, acquisitions


def sample_coverage(qualified, industry_participations):
    """Summarize qualified investors, companies, industries, and screening counts."""
    audit = qualified.attrs["audit"]
    return pd.DataFrame([{
        "country": qualified.country.iloc[0],
        **audit,
        "mapped_participations": len(industry_participations),
        "investors": industry_participations.investor_id.nunique(),
        "companies": industry_participations.company_id.nunique(),
        "industries": industry_participations.naics4.nunique(),
    }])




def main():
    """Screen country-specific raw records and save qualified participations."""
    import json
    from ..settings import OUT_PREP
    OUT_PREP.mkdir(parents=True, exist_ok=True)
    events, *_ = read_records()
    events.to_csv(OUT_PREP / "screened_participations.csv", index=False)
    (OUT_PREP / "screening_audit.json").write_text(
        json.dumps(events.attrs["audit"], indent=2), encoding="utf-8")
    print(pd.Series(events.attrs["audit"], name="screening").to_string())


if __name__ == "__main__":
    main()
