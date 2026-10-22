"""
Train SmartCart behavior-first segmentation.
Model inputs exclude marital status, education, demographics, and campaign Response.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from preprocessing import preprocess_pipeline, MODEL_FEATURES

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "smartcart_customers.csv"
MODEL_DIR = BASE / "model"
RANDOM_STATE = 42
PCA_VARIANCE_TARGET = 0.80


def find_optimal_k(X, k_range=range(2, 11)):
    inertias, scores = [], []
    all_k = list(range(1, 11))
    for k in all_k:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        if k >= 2:
            scores.append({"k": k, "silhouette": float(silhouette_score(X, labels))})
    # Geometric knee estimate: maximum distance from the normalized end-point chord.
    # It avoids an extra dependency and is reported as a heuristic, not ground truth.
    x = np.asarray(all_k, dtype=float)
    y = np.asarray(inertias, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    distance = np.abs((1 - xn) * (yn[0] - yn[-1]) - (1 - yn) * (xn[0] - xn[-1]))
    knee = int(x[np.argmax(distance)])
    return inertias, knee, pd.DataFrame(scores)


def make_persona_names(df, cluster_col="Cluster"):
    """Rule-based, interpretable names based on behavior—not demographics."""
    p = df.groupby(cluster_col).agg(
        spend=("Total_Spending", "median"),
        purchases=("Total_Purchases", "median"),
        recency=("Recency", "median"),
        deal_share=("Deal_Purchase_Share", "median"),
        online_share=("Online_Purchase_Share", "median"),
        web_visits=("NumWebVisitsMonth", "median"),
    )
    spend_mid = p["spend"].median()
    freq_mid = p["purchases"].median()
    recency_mid = p["recency"].median()
    deal_mid = p["deal_share"].median()
    online_mid = p["online_share"].median()
    visits_mid = p["web_visits"].median()
    names = {}
    for cid, r in p.iterrows():
        if r["recency"] >= recency_mid * 1.25:
            base = "At-Risk / Inactive"
        elif r["spend"] >= spend_mid and r["purchases"] >= freq_mid:
            if r["online_share"] >= online_mid:
                base = "High-Value Digital Active"
            else:
                base = "High-Value Store/Catalog Active"
        elif r["deal_share"] >= deal_mid and r["purchases"] >= freq_mid:
            base = "Deal-Driven Regulars"
        elif r["online_share"] >= online_mid and r["web_visits"] >= visits_mid:
            base = "Digital-First Researchers"
        elif r["spend"] >= spend_mid:
            base = "High-Spend Occasional"
        else:
            base = "Low-Engagement"
        names[int(cid)] = base
    # Make labels unique if two clusters satisfy the same behavior rule.
    used = {}
    for cid in sorted(names):
        label = names[cid]
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            names[cid] = f"{label} (Group {used[label]})"
    return names, p


def plot_k_selection(inertias, metrics_df, pca_variance, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(range(1, len(inertias)+1), inertias, marker="o")
    axes[0].set_title("Elbow Method (WCSS)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("WCSS / inertia")
    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o")
    axes[1].axvline(4, linestyle="--", alpha=.6, label="Reference k=4")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[2].bar(range(1, len(pca_variance)+1), np.cumsum(pca_variance))
    axes[2].axhline(PCA_VARIANCE_TARGET, linestyle="--", label="80% target")
    axes[2].set_title("Cumulative PCA Variance")
    axes[2].set_xlabel("Number of components")
    axes[2].set_ylabel("Cumulative explained variance")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    df, features, X_scaled, scaler, row_counts = preprocess_pipeline(raw)

    # Keep enough components to explain at least 80% of the variance.
    pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver="full", random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    inertias, elbow_k, metrics = find_optimal_k(X_pca)
    # Select k=4 as the documented business choice when elbow is ambiguous;
    # silhouette is reported separately and is not treated as the only criterion.
    # Keep k=4 as the project's business-readable baseline, but expose the
    # geometric elbow estimate and candidate silhouette scores for honest review.
    final_k = 4

    kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=30)
    labels = kmeans.fit_predict(X_pca)
    df["Cluster"] = labels
    names, profile = make_persona_names(df)
    df["Segment"] = df["Cluster"].map(names)

    ag = AgglomerativeClustering(n_clusters=final_k)
    ag_labels = ag.fit_predict(X_pca)
    comparison = {
        "kmeans_silhouette": round(float(silhouette_score(X_pca, labels)), 3),
        "agglomerative_silhouette": round(float(silhouette_score(X_pca, ag_labels)), 3),
        "kmeans_davies_bouldin": round(float(davies_bouldin_score(X_pca, labels)), 3),
        "agglomerative_davies_bouldin": round(float(davies_bouldin_score(X_pca, ag_labels)), 3),
        "selected_k": final_k,
        "geometric_elbow_estimate": int(elbow_k) if elbow_k is not None else None,
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "pca_variance_target": PCA_VARIANCE_TARGET,
        "model_features": MODEL_FEATURES,
        "selection_note": "k=4 retained as a business-readable baseline; elbow is considered alongside silhouette and interpretability.",
    }

    # SHAP-compatible surrogate: explains features associated with learned cluster labels,
    # not the K-Means algorithm itself. Keep only model inputs.
    surrogate = RandomForestClassifier(n_estimators=250, max_depth=7, random_state=RANDOM_STATE)
    surrogate.fit(features, labels)

    df.to_csv(MODEL_DIR / "segmented_customers.csv", index=True, index_label="ID")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(pca, MODEL_DIR / "pca.pkl")
    joblib.dump(kmeans, MODEL_DIR / "kmeans_model.pkl")
    joblib.dump(names, MODEL_DIR / "cluster_names.pkl")
    joblib.dump(surrogate, MODEL_DIR / "surrogate_model.pkl")
    joblib.dump(comparison, MODEL_DIR / "agglomerative_comparison.pkl")
    joblib.dump(MODEL_FEATURES, MODEL_DIR / "feature_cols.pkl")
    joblib.dump(profile, MODEL_DIR / "cluster_behavior_profile.pkl")
    metrics.to_csv(MODEL_DIR / "k_selection_metrics.csv", index=False)
    plot_k_selection(inertias, metrics, pca.explained_variance_ratio_,
                     MODEL_DIR / "k_selection.png")
    (MODEL_DIR / "training_summary.json").write_text(json.dumps({
        "raw_records": len(raw), "records_after_filter": len(df),
        "records_removed": len(raw) - len(df), "cluster_counts": df["Segment"].value_counts().to_dict(),
        "metrics": comparison, "segment_names": names,
        "geometric_elbow_estimate": int(elbow_k) if elbow_k is not None else None,
    }, indent=2), encoding="utf-8")

    print(f"Raw rows: {len(raw)}; retained rows: {len(df)}; removed: {len(raw)-len(df)}")
    print(f"PCA components: {pca.n_components_}; explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print("Cluster counts:", df["Segment"].value_counts().to_dict())
    print("Metrics:", comparison)


if __name__ == "__main__":
    main()
