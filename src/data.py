import pandas as pd
from .config import DATA_RAW, DATA_PROCESSED, COLUMNS, TARGET

def load_raw(path=DATA_RAW):
    return pd.read_csv(path, header=None, names=COLUMNS)

def audit_data(df):
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "predictors": int(df.shape[1]-1),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "classes": int(df[TARGET].nunique()),
        "class_counts": {str(k): int(v) for k,v in df[TARGET].value_counts().sort_index().items()},
    }

def clean_data(df):
    out=df.drop_duplicates().dropna().copy()
    for c in COLUMNS:
        out[c]=pd.to_numeric(out[c], errors="raise")
    out[TARGET]=out[TARGET].astype(int)
    return out

def save_processed(df, path=DATA_PROCESSED):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
