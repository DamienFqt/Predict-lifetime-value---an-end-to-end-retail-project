import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERMEDIATE_DATA_DIR

def preprocess_sales_data_df(sales_df: pd.DataFrame, clv_df: pd.DataFrame, y_analysis: int = 2024) -> pd.DataFrame:
    """
    Prétraite les données de ventes et CLV pour l'année d'analyse.
    Retire les clients plus actifs l'année d'analyse, retire les clients sans achat l'année d'analyse
    Args:
        sales_df: DataFrame des ventes
        clv_df: DataFrame des CLV
        y_analysis: année d'analyse
    """
    sales = sales_df[sales_df["Year"] == y_analysis]
    merged = sales.merge(clv_df, on="Client_ID", how="left")
    merged = merged[merged["Recency"].notna()]
    merged["avg_purchase"] = np.where(merged["Nb_Purchases"] > 0,
                                  merged["Tot_Purchases"] / merged["Nb_Purchases"],
                                  0)
    # Features selection
    merged=merged[["Nb_Purchases","avg_purchase","Recency","Tenure","CLV"]] # 
    return merged


def preprocess_sales_data_files(
    sales_path: Path,
    clv_path: Path,
    y_analysis: int = 2024,
    output_name: str = "processed_sales_2024.csv"
) -> pd.DataFrame:
    """
    Prétraite les fichiers CSV et exporte le résultat.

    Args:
        sales_path: chemin du fichier CSV ventes
        clv_path: chemin du fichier CSV CLV
        y_analysis: année d'analyse
        output_name: nom du fichier de sortie (optionnel)
    """
    sales_df = pd.read_csv(sales_path)
    clv_df = pd.read_csv(clv_path)

    preprocessed = preprocess_sales_data_df(sales_df, clv_df, y_analysis)

    # Définir le nom du fichier de sortie si non fourni
    if output_name is None:
        output_name = f"preprocessed_sales_{y_analysis}.csv"

    # Assurer que le dossier existe
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Export du fichier
    output_file = PROCESSED_DATA_DIR / output_name
    preprocessed.to_csv(output_file, index=False)
    return preprocessed


if __name__ == "__main__":
    print("Preprocessing sales data...")

    # Chemins vers les CSV existants
    sales_path = RAW_DATA_DIR / "sales.csv"
    clv_path = INTERMEDIATE_DATA_DIR / "CLV_data.csv"

    # Appel de la fonction de prétraitement avec export
    preprocessed_df = preprocess_sales_data_files(
        sales_path=sales_path,
        clv_path=clv_path,
        y_analysis=2024
    )

    print("Preprocessing completed. File saved to:", PROCESSED_DATA_DIR)
