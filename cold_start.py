import pandas as pd

PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]


def classify_customer_status(customer_id, df_encoded: pd.DataFrame, tenure_threshold_days: int = 30, min_purchases: int = 2) -> str:
    if customer_id not in df_encoded.index:
        return "unknown_id"
    row = df_encoded.loc[customer_id]
    total_purchases = sum(row.get(c, 0) for c in PURCHASE_COLS)
    if row["Customer_tenure_days"] <= tenure_threshold_days and total_purchases < min_purchases:
        return "new_customer"
    return "established"
