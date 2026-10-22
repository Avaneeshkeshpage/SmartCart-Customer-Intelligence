import shap
import pandas as pd


def get_customer_explanation(surrogate_model, customer_row: pd.Series, feature_cols: list, cluster_id: int, top_n: int = 6):
    explainer = shap.TreeExplainer(surrogate_model)
    X_customer = customer_row[feature_cols].values.reshape(1, -1)
    shap_values = explainer.shap_values(X_customer)
    if isinstance(shap_values, list):
        contributions = shap_values[cluster_id][0]
    else:
        contributions = shap_values[0, :, cluster_id]
    result = pd.DataFrame({
        "Feature": feature_cols,
        "Value": customer_row[feature_cols].values,
        "Contribution": contributions,
    }).sort_values("Contribution", key=abs, ascending=False)
    return result.head(top_n)
