from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "Wine.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "wine_clean.csv"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
RANDOM_STATE = 2026
TARGET = "cultivar"
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280_od315", "proline"
]
COLUMNS = [TARGET] + FEATURES
