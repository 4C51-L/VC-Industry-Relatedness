from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "database"
RAW = DATABASE
SUPPLEMENT = DATABASE
OUT_PREP = ROOT / "result" / "preparation"
OUT_SPACE = ROOT / "result" / "industry_space"
OUT_STRUCTURE = ROOT / "result" / "structural_validation"
OUT_TRADE = ROOT / "result" / "external_validation"
OUT_KBV = ROOT / "result" / "kbv"
OUT_SI = ROOT / "result" / "si"
OUT_NULL = ROOT / "result" / "null_model"

PERIODS = ((1990, 1994), (1995, 1999), (2000, 2004),
           (2005, 2009), (2010, 2014), (2015, 2020))
ALPHA = 0.1
SEED = 42
EMBEDDING_DIM = 128
WALK_LENGTH = 30
WALKS_PER_NODE = 200
WINDOW = 5
EPOCHS = 5
N_CLUSTERS = 5
