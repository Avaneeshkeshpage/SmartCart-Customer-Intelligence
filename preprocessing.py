"""
Behavior-first feature engineering for SmartCart.
Demographic labels and campaign outcome Response are intentionally excluded
from clustering inputs to reduce proxy segmentation and target leakage.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
DEAL_COL = "NumDealsPurchases"
SPENDING_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]
MODEL_FEATURES = [
    "Recency", "Total_Spending", "Total_Purchases",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumDealsPurchases", "NumWebVisitsMonth",
    "Online_Purchase_Share", "Deal_Purchase_Share",
    "Web_Visit_to_Purchase_Ratio"
]


def engineer_features(raw_df: pd.DataFrame, reference_year: int = None) -> pd.DataFrame:
    df = raw_df.copy()
    df["Income"] = df["Income"].fillna(df["Income"].median())
    dt = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    ref_year = reference_year if reference_year is not None else int(dt.max().year + 1)
    df["Age"] = ref_year - df["Year_Birth"]
    df["Customer_tenure_days"] = (dt.max() - dt).dt.days.fillna(0).astype(int)
    df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
    df["Total_Purchases"] = df[PURCHASE_COLS + [DEAL_COL]].sum(axis=1)
    channel_purchases = df[PURCHASE_COLS].sum(axis=1)
    df["Online_Purchase_Share"] = df["NumWebPurchases"] / channel_purchases.replace(0, np.nan)
    df["Online_Purchase_Share"] = df["Online_Purchase_Share"].fillna(0).clip(0, 1)
    df["Deal_Purchase_Share"] = df[DEAL_COL] / df["Total_Purchases"].replace(0, np.nan)
    df["Deal_Purchase_Share"] = df["Deal_Purchase_Share"].fillna(0).clip(0, 1)
    df["Web_Visit_to_Purchase_Ratio"] = (
        df["NumWebVisitsMonth"] / (df["NumWebPurchases"] + 1)
    )
    return df


def preprocess_pipeline(raw_df: pd.DataFrame, reference_year: int = None):
    """Return enriched customer table, scaled behavior features, scaler, and row counts."""
    df = engineer_features(raw_df, reference_year=reference_year)
    before = len(df)
    # Keep robust, transparent outlier filters; do not drop a customer for missing income
    # after median imputation. The age filter also removes malformed birth-year records.
    df = df[(df["Income"] < 600_000) & (df["Age"].between(18, 100))].copy()
    after = len(df)
    df = df.drop_duplicates(subset=["ID"], keep="first").set_index("ID")
    features = df[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return df, features, X_scaled, scaler, (before, after)


def transform_customer_features(customer: dict, scaler):
    """Scale a manually entered behavioral profile using the training scaler."""
    import pandas as pd
    row = pd.DataFrame([{name: customer.get(name, 0) for name in MODEL_FEATURES}])
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0)
    return scaler.transform(row[MODEL_FEATURES])
