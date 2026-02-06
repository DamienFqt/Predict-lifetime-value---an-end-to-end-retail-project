import time
import numpy as np
from src.pipelines.generate_data import generate_data
from src.pipelines.compute_clv import compute_clv, save_clv
from src.models.v1.preprocess import preprocess_sales_data_files
from src.config import RAW_DATA_DIR, INTERMEDIATE_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.models.v1.train import train_rf_model
from src.models.v1.evaluate import evaluate_model


def main():
    try:
        # =======================
        # Génération des données
        # =======================
        print("Génération des données...")
        start_gen = time.time()

        sales_df = generate_data(
            n=150000,
            T=26,
            mu_unit=np.log(50) - 0.32,
            sigma_unit=0.8,
            y_start=2020,
            output_dir=RAW_DATA_DIR,
            output_name="sales.csv"
        )

        print(f"Données générées en {time.time() - start_gen:.2f} secondes")
        print(sales_df.info())
        print(sales_df.head())
        print(sales_df[["Nb_Purchases","Tot_Purchases"]].describe())
        avg_purchase = sales_df["Tot_Purchases"] / sales_df["Nb_Purchases"].replace(0, 1)
        print("Résumé numérique de avg_purchase:", avg_purchase.describe())
 

        # =======================
        # Calcul CLV
        # =======================
        print("Calcul des CLV...")
        clv_df = compute_clv(sales_df)
        save_clv(clv_df, INTERMEDIATE_DATA_DIR, "CLV_data.csv")
        print("CLV sauvegardées")
        print(clv_df.describe())
        # =======================
        # Préprocessing
        # =======================
        print("Préprocessing...")
        preprocessed_data = preprocess_sales_data_files(
            sales_path=RAW_DATA_DIR / "sales.csv",
            clv_path=INTERMEDIATE_DATA_DIR / "CLV_data.csv",
            y_analysis=2024,
            output_name="processed_sales_2024.csv"
        )
        print("Préprocessing terminé")

        # =======================
        # Training
        # =======================
        print("Entraînement du modèle v1 (RandomForest / CLV)...")

        model_path = train_rf_model(
            csv_name="processed_sales_2024.csv",
            version="v1",
            target_col="CLV",
            algo_name="rf",
            n_estimators=100,
            test_size=0.2
        )

        print("Modèle entraîné :", model_path)

        # =======================
        # Évaluation
        # =======================
        print("Évaluation du modèle...")
        evaluate_model(
            model_path=model_path,
            csv_name="processed_sales_2024.csv",
            target_col="CLV"
        )

        print("Pipeline terminé avec succès")

    except Exception as e:
        print("Erreur :", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=== Début main.py ===")
    start_total = time.time()
    main()
    print(f"=== Fin main.py ({time.time() - start_total:.2f} s) ===")
