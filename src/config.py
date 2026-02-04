from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "src" / "models"


def debug():
    print("BASE_DIR (racine projet) =", BASE_DIR)
    print("DATA_DIR =", DATA_DIR)
    print("RAW_DATA_DIR =", RAW_DATA_DIR)
    print("INTERMEDIATE_DATA_DIR =", INTERMEDIATE_DATA_DIR)
    print("PROCESSED_DATA_DIR =", PROCESSED_DATA_DIR)

if __name__ == "__main__":
    debug()

