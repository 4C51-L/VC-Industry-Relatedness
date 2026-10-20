from __future__ import annotations
import re
import pandas as pd
from ..settings import RAW


def normalize_name(value: str) -> str:
    """Normalize an investor name for case-insensitive exact matching."""
    return value.strip().casefold()

def parse_investor_list(
    value: str, known_names: set[str]
) -> tuple[list[tuple[str, str]], int]:
    """Parse a comma-separated list with longest known-name matching."""
    text = value.strip()
    if not text:
        return [], 0

    matched: list[tuple[str, str]] = []
    unresolved_fragments = 0
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        while cursor < text_length and text[cursor].isspace():
            cursor += 1
        if cursor >= text_length:
            break

        comma_positions: list[int] = []
        comma = text.find(",", cursor)
        while comma != -1:
            comma_positions.append(comma)
            comma = text.find(",", comma + 1)

        match_end: int | None = None
        match_display = ""
        match_key = ""
        for candidate_end in reversed([*comma_positions, text_length]):
            candidate = text[cursor:candidate_end].strip()
            candidate_key = normalize_name(candidate)
            if candidate_key and candidate_key in known_names:
                match_end = candidate_end
                match_display = candidate
                match_key = candidate_key
                break

        if match_end is not None:
            matched.append((match_display, match_key))
            cursor = match_end
        else:
            next_comma = text.find(",", cursor)
            fragment_end = next_comma if next_comma != -1 else text_length
            if text[cursor:fragment_end].strip():
                unresolved_fragments += 1
            cursor = fragment_end

        if cursor < text_length and text[cursor] == ",":
            cursor += 1

    return matched, unresolved_fragments

STATES = frozenset(x.replace('_', ' ') for x in (
    'Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada Oregon Ohio Oklahoma Pennsylvania Tennessee Texas Utah Vermont Virginia Washington Wisconsin Wyoming '
    'New_Hampshire New_Jersey New_Mexico New_York North_Carolina North_Dakota Rhode_Island South_Carolina South_Dakota West_Virginia District_of_Columbia'
).split())

def state_from_location(value: str) -> tuple[str, str]:
    location = value.strip()
    if not location:
        return '', 'empty'
    parts = [p.strip() for p in location.split(',')]
    if parts[-1] != 'United States':
        return '', 'foreign'
    if len(parts) < 2 or parts[-2] not in STATES:
        return '', 'invalid_us_state'
    return parts[-2], 'us_state'


def _year(value):
    return pd.to_datetime(value, errors="coerce").dt.year

def _investor_kind(value):
    labels = {x.strip().lower() for x in re.split(r"[,;|]", str(value))}
    return bool(labels & {"micro vc", "venture capital", "corporate venture capital"})

def _window(df):
    return df.loc[df.year.between(1990, 2020)].copy()

def _exclude_rounds(df):
    return df.loc[~df.round_type.fillna("").str.contains(
        r"angel|post[- ]?ipo", case=False, regex=True)].copy()

def _qualify(df):
    df = df.drop_duplicates(["country", "event_id", "investor_id"])
    n = df.groupby("investor_id").event_id.transform("size")
    return df.loc[n >= 10].copy()

def read_records():
    rounds = pd.read_csv(RAW / "funding_rounds.csv", low_memory=False)
    raw_rounds = len(rounds)
    investors = pd.read_csv(RAW / "investors.csv", low_memory=False)
    companies = pd.read_csv(RAW / "companies.csv", low_memory=False)
    acquisitions = pd.read_csv(RAW / "acquisitions.csv", low_memory=False)
    rounds = rounds.rename(columns={"Transaction Name URL": "event_id",
        "Organization Name URL": "company_id", "Announced Date": "event_date",
        "Funding Type": "round_type", "Investor Names": "investor_names",
        "Lead Investors": "lead_names", "Organization Industries": "industry_label",
        "Organization Location": "company_location"})
    inv = investors.rename(columns={"Organization/Person Name": "investor_name",
        "Organization/Person Name URL": "investor_id", "Investor Type": "investor_type",
        "Location": "region"})
    inv = inv.loc[inv.investor_type.map(_investor_kind),
                  ["investor_name", "investor_id", "region"]].drop_duplicates("investor_id")
    inv["investor_key"] = inv.investor_name.map(normalize_name)
    inv["region"] = inv.region.fillna("").map(lambda value: state_from_location(value)[0])
    inv = inv.loc[inv.region.ne("")]
    rounds["year"] = _year(rounds.event_date)
    rounds = _exclude_rounds(_window(rounds))
    rounds = rounds.loc[rounds.company_location.fillna("").str.contains(
        r"United States|\bUSA\b", case=False, regex=True)]
    known_names = {normalize_name(name) for name in inv.investor_name.dropna()}
    rows = rounds.assign(investor_key=rounds.investor_names.fillna("").map(
        lambda value: [key for _, key in parse_investor_list(value, known_names)[0]])).explode("investor_key")
    rows = rows.merge(inv, on="investor_key", how="inner")
    rows["is_lead"] = rows.apply(lambda r: r.investor_key in
        {key for _, key in parse_investor_list(str(r.lead_names), known_names)[0]}, axis=1)
    rows["country"] = "US"
    rows = rows.dropna(subset=["event_id", "company_id", "investor_id", "industry_label"])
    for field in ("event_id", "company_id", "investor_id", "industry_label"):
        rows[field] = rows[field].astype("string").str.strip()
        rows = rows.loc[rows[field].ne("") & rows[field].ne("nan")]
    screened = len(rows)
    rows = _qualify(rows)
    rows.attrs["audit"] = {"raw_source_rows": raw_rounds,
                           "screened_participations": screened,
                           "qualified_participations": len(rows)}
    company = companies.rename(columns={"Organization Name URL": "company_id",
        "IPO Date": "ipo_date", "Exit Date": "exit_date"})
    acquisition = acquisitions.rename(columns={"Transaction Name URL": "acquisition_id",
        "Acquiree Name URL": "company_id", "Announced Date": "acquisition_date"})
    return rows, company, acquisition


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
