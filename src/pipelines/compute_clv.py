import pandas as pd
from pathlib import Path
from src.config import INTERMEDIATE_DATA_DIR, RAW_DATA_DIR

def compute_clv(final_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Customer Lifetime Value (CLV) per client.
    """
    required_cols = {"Client_ID", "Tot_Purchases"}
    missing = required_cols - set(final_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return (
        final_data
        .groupby("Client_ID", as_index=True)["Tot_Purchases"]
        .sum()
        .to_frame(name="CLV")
    )


def save_clv(
    clv_df: pd.DataFrame,
    output_dir: Path = INTERMEDIATE_DATA_DIR,
    output_name: str = "CLV_data.csv"
):
    """
    Save CLV DataFrame to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clv_df.to_csv(output_dir / output_name)


# =====================
# CLI / debug block
# =====================
if __name__ == "__main__":
    print("Computing labels...")
    input_file = RAW_DATA_DIR / "sales.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found. Generate data first.")

    final_data = pd.read_csv(input_file)

    clv_df = compute_clv(final_data)
    save_clv(clv_df)

    print("Labels computation completed.")
