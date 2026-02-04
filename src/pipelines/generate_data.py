import numpy as np
import pandas as pd
import scipy
from datetime import datetime, timedelta
from src.config import DATA_DIR, RAW_DATA_DIR, BASE_DIR
import os

def generate_data(
    n: int = 15000,
    T: int = 26,
    mu_unit: float = np.log(50) - 0.32,
    sigma_unit: float = 0.8,
    y_start: int = 2020,
    output_dir: str = RAW_DATA_DIR,
    output_name: str = "sales.csv"
):
    """
    Génère un DataFrame : données d'achats clients complètes sur plusieurs années (panel_data).

    Args:
        n (int): Nombre de clients.
        T (int): Nombre d'années simulées.
        mu_unit (float) : Nombre d'achats moyen souhaité (paramètre log-normal achat moyen)
        sigma_unit (float) : Dispersion des achats moyens (paramètre log-normal achat moyen)
        y_start (int): Année de début de la simulation.
        output_dir (str): Dossier de sortie pour les fichiers CSV, à partir du dossier courant
    """
    # =====================
    output_dir.mkdir(parents=True, exist_ok=True)  

    # =====================
    # Parameter s
    # =====================
    y_end = y_start + T
    y_current = pd.Timestamp.now().year

    np.random.seed(42)
    shape = 2.0
    scale = 1.0
    sigma_satisfaction = 0.5
    a_value = mu_unit
    b_value = 0.3

    # =====================
    # Client IDs
    # =====================
    client_ids = np.array([f"C-{i:05d}" for i in range(1, n + 1)])

    # =====================
    # Latent heterogeneity
    # =====================
    lambda_clients = np.random.gamma(shape=shape, scale=scale, size=n)

    # =====================
    # Storage
    # =====================
    all_purchases = {client_id: {} for client_id in client_ids}
    records = []

    # =====================
    # Simulation
    # =====================
    for year in range(y_start, y_end):
        # Frequency
        nb_purchases = np.random.poisson(lam=lambda_clients, size=n)

        # Satisfaction
        satisfaction_raw = nb_purchases * np.random.uniform(0, 5, size=n)
        yearly_satisfaction = np.minimum(satisfaction_raw, 5)

        # Basket value depends on satisfaction
        avg_purchase = np.random.lognormal(
            mean=a_value + b_value * yearly_satisfaction,
            sigma=sigma_unit,
            size=n
        )

        # Total value
        tot_purchases = nb_purchases * avg_purchase

        # Purchase dates
        for i in range(n):
            client_id = client_ids[i]
            if nb_purchases[i] > 0:
                all_days = pd.date_range(
                    start=f"{year}-01-01",
                    end=f"{year}-12-31",
                    freq="D"
                )
                dates = np.random.choice(
                    all_days,
                    size=nb_purchases[i],
                    replace=True
                )
                all_purchases[client_id][year] = dates
            else:
                all_purchases[client_id][year] = np.array([])

        # Churn
        churn = np.random.binomial(
            1,
            1 / (1 + 8 * yearly_satisfaction),
            size=n
        )

        # Panel slice
        year_df = pd.DataFrame({
            "Client_ID": client_ids,
            "Year": year,
            "Nb_Purchases": nb_purchases,
            "Tot_Purchases": tot_purchases,
            "Yearly_Satisfaction": yearly_satisfaction,
            "Churn": churn,
        })

        records.append(year_df)

    # =====================
    # Build panel
    # =====================
    panel_data = pd.concat(records, ignore_index=True)

    # =====================
    # Build long purchase table
    # =====================
    rows = []
    for client_id, years in all_purchases.items():
        for year, dates in years.items():
            for d in dates:
                rows.append([client_id, year, pd.Timestamp(d)])

    purchases_df = pd.DataFrame(
        rows,
        columns=["Client_ID", "Year", "Purchase_Date"]
    )

    # =====================
    # Aggregates
    # =====================
    last = purchases_df.groupby(["Client_ID", "Year"])["Purchase_Date"].max()
    first = purchases_df.groupby(["Client_ID", "Year"])["Purchase_Date"].min()
    first_alltime = purchases_df.groupby("Client_ID")["Purchase_Date"].min()

    # Shift first purchase of each year to previous year
    first_next = (
        first
        .reset_index()
        .assign(Year=lambda x: x["Year"] - 1)
        .rename(columns={"Purchase_Date": "First_Purchase_Next"})
    )

    # =====================
    # Merge into panel
    # =====================
    panel_data = (
        panel_data
        .merge(
            last.reset_index().rename(columns={"Purchase_Date": "Last_Purchase"}),
            on=["Client_ID", "Year"],
            how="left"
        )
        .merge(
            first_next,
            on=["Client_ID", "Year"],
            how="left"
        )
        .merge(
            first_alltime.reset_index().rename(
                columns={"Purchase_Date": "First_Purchase_AllTime"}
            ),
            on="Client_ID",
            how="left"
        )
    )

    panel_data.loc[panel_data["Churn"] == 1, "First_Purchase_Next"] = pd.NaT

    # =====================
    # Recency
    # =====================
    fallback = pd.to_datetime(panel_data["Year"] + 2, format="%Y")
    panel_data["Recency"] = (
        panel_data["First_Purchase_Next"].fillna(fallback)
        - panel_data["Last_Purchase"]
    ).dt.days

    # =====================
    # Tenure
    # =====================
    tenure_ref = pd.to_datetime(panel_data["Year"] + 1, format="%Y")
    panel_data["Tenure"] = (
        tenure_ref - panel_data["First_Purchase_AllTime"]
    ).dt.days

    # =====================
    # Drop last year
    # =====================
    panel_data = panel_data[panel_data["Year"] < y_end - 1]

    # =====================
    # Remove data after churn
    # =====================
    panel_data = panel_data.sort_values(["Client_ID", "Year"])
    panel_data["Has_Churned"] = (
        panel_data
        .groupby("Client_ID")["Churn"]
        .cumsum()
    )
    panel_data["Period_since_churn"] = (
        panel_data
        .groupby("Client_ID")["Has_Churned"]
        .cumsum()
    )

    final_data = (
        panel_data[panel_data["Period_since_churn"] <= 1]
        .drop(columns=["Has_Churned", "Period_since_churn"])
    )

    # =====================
    # Logs
    # =====================
    print(f"Taille de final_data : {final_data.shape}")

    # =====================
    # Export
    # =====================
    final_data.to_csv(output_dir / output_name, index=False)
    print("BASE_DIR =", BASE_DIR)
    print("OUTPUT_DIR =", output_dir)
    print("chemin final :", os.path.join(output_dir, output_name))


    return final_data

if __name__ == "__main__":
    print("Generating data...")
    generate_data()
    print("Data generation completed.")
