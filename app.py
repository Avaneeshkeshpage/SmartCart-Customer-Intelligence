import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from marketing_actions import get_action, get_new_customer_action
from explainability import get_customer_explanation
from cold_start import classify_customer_status
from sklearn.cluster import DBSCAN
from auth import require_login
from database import load_customers as load_customers_from_db, save_profile_assignment, list_profile_assignments

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = MODEL_DIR / "segmented_customers.csv"

st.set_page_config(
    page_title="SmartCart | Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 14px 16px; border-radius: 14px;
    }
    .small-muted {opacity: 0.75; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    required = [
        "kmeans_model.pkl", "scaler.pkl", "pca.pkl",
        "cluster_names.pkl", "surrogate_model.pkl",
        "agglomerative_comparison.pkl", "feature_cols.pkl",
    ]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): " + ", ".join(missing)
            + ". Run `python train_model.py` from the project folder."
        )
    return {
        "kmeans": joblib.load(MODEL_DIR / "kmeans_model.pkl"),
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "pca": joblib.load(MODEL_DIR / "pca.pkl"),
        "cluster_names": joblib.load(MODEL_DIR / "cluster_names.pkl"),
        "surrogate": joblib.load(MODEL_DIR / "surrogate_model.pkl"),
        "comparison": joblib.load(MODEL_DIR / "agglomerative_comparison.pkl"),
        "feature_cols": joblib.load(MODEL_DIR / "feature_cols.pkl"),
    }


@st.cache_data
def load_customers():
    """Read customer records from SQLite; seed once from the bundled CSV."""
    data = load_customers_from_db()
    if data.empty:
        raise RuntimeError("No customer records found. Check model/segmented_customers.csv and smartcart.db.")
    data.index = data.index.map(str)
    data.index.name = "Customer ID"
    return data


def safe_float(value, digits=3):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def get_profile(df):
    cols = [
        c for c in [
            "Income", "Total_Spending", "Recency", "Age",
            "Customer_tenure_days", "NumWebPurchases",
            "NumStorePurchases", "NumDealsPurchases",
            "NumWebVisitsMonth", "Response"
        ] if c in df.columns
    ]
    profile = df.groupby(["Cluster", "Segment"])[cols].mean().round(1)
    profile["Customers"] = df.groupby(["Cluster", "Segment"]).size()
    if "Response" in profile.columns:
        profile["Campaign Response (%)"] = (profile["Response"] * 100).round(1)
        profile = profile.drop(columns=["Response"])
    return profile.reset_index()


def render_customer_details(customer_id, df, artifacts):
    if customer_id not in df.index:
        st.error(f"Customer ID **{customer_id}** was not found in the segmented dataset.")
        return

    row = df.loc[customer_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    st.success(f"Customer **{customer_id}** found")
    a, b, c, d = st.columns(4)
    a.metric("Assigned cluster", str(int(row["Cluster"])))
    b.metric("Segment", str(row["Segment"]))
    b.caption("Segment names are inferred from cluster profiles.")
    c.metric("Income", f"{row['Income']:,.0f}" if "Income" in row and pd.notna(row["Income"]) else "N/A")
    d.metric("Total spending", f"{row['Total_Spending']:,.0f}" if "Total_Spending" in row and pd.notna(row["Total_Spending"]) else "N/A")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Customer details")
        preferred = [
            "Age", "Recency", "Customer_tenure_days",
            "NumWebPurchases", "NumCatalogPurchases",
            "NumStorePurchases", "NumDealsPurchases",
            "NumWebVisitsMonth", "Response", "Complain"
        ]
        details = {key: row[key] for key in preferred if key in row.index}
        if details:
            st.dataframe(
                pd.DataFrame({"Feature": list(details.keys()), "Value": list(details.values())}),
                hide_index=True, use_container_width=True
            )
    with right:
        st.markdown("#### Suggested marketing action")
        try:
            action = get_action(row["Segment"])
            st.write(f"**Profile:** {action.get('description', 'No profile description available.')}")
            st.info(action.get("action", "Review this customer's profile before selecting an action."))
        except Exception:
            st.info("Review this customer's profile to decide on an appropriate action.")

    with st.expander("Why was this customer assigned to this segment?"):
        try:
            explanation = get_customer_explanation(
                artifacts["surrogate"], row, artifacts["feature_cols"], int(row["Cluster"])
            )
            if explanation is not None and len(explanation):
                fig, ax = plt.subplots(figsize=(7, 3))
                vals = explanation["Contribution"].astype(float)
                colors = ["#168f69" if v >= 0 else "#d95f45" for v in vals]
                ax.barh(explanation["Feature"], vals, color=colors)
                ax.axvline(0, color="gray", linewidth=0.8)
                ax.set_xlabel("Contribution to the surrogate model's cluster prediction")
                ax.invert_yaxis()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("This explains a Random Forest surrogate trained to mimic cluster assignments; it is not an explanation of K-Means internals or a causal explanation.")
            else:
                st.info("No explanation values were returned for this customer.")
        except Exception as exc:
            st.warning(f"Explanation is unavailable for this customer: {exc}")


def main():
    if not require_login():
        st.stop()
    st.title("🛒 SmartCart Customer Intelligence")
    st.caption("Behavior-first customer intelligence: RFM-style features, interpretable segments, customer lookup, and new-profile assignment.")

    try:
        artifacts = load_artifacts()
        df = load_customers()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    required_cols = {"Cluster", "Segment", "Income", "Total_Spending"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        st.error("Segmented customer file is missing columns: " + ", ".join(sorted(missing_cols)))
        st.stop()

    # Sidebar controls
    st.sidebar.title("SmartCart Controls")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Search Customer", "Explore Customers", "Segment Profiles", "Model Evaluation", "DBSCAN Outlier Detection", "Predict New Customer", "Assignment History"]
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()
    segments = sorted(df["Segment"].dropna().unique().tolist())
    selected_segments = st.sidebar.multiselect(
        "Segments to include", options=segments, default=segments
    )
    st.sidebar.caption(f"{len(df):,} customer records loaded")
    st.sidebar.download_button(
        "⬇️ Download full segmented data",
        data=df.to_csv(index=True).encode("utf-8"),
        file_name="smartcart_segmented_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )

    view = df[df["Segment"].isin(selected_segments)].copy()
    # Handle the case where the user deselects every segment
    if not selected_segments:
        st.info("No segments selected. Select at least one segment from the sidebar to view customer data.")
        st.stop()

    if page == "Overview":
        st.subheader("Business overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers in view", f"{len(view):,}")
        m2.metric("Segments", f"{view['Segment'].nunique():,}")
        m3.metric("Average spend", f"{view['Total_Spending'].mean():,.1f}" if len(view) else "N/A")
        m4.metric("Average income", f"{view['Income'].mean():,.0f}" if len(view) else "N/A")

        st.markdown("### Segment distribution")
        if len(view):
            counts = view["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
            c1, c2 = st.columns([1, 1.3])
            with c1:
                st.dataframe(counts, hide_index=True, use_container_width=True)
            with c2:
                st.bar_chart(counts.set_index("Segment")["Customers"], use_container_width=True)

        st.markdown("### Customer map")
        st.caption("Each dot represents a customer projected into two dimensions by the saved PCA model.")
        try:
            feature_cols = artifacts["feature_cols"]
            usable = [c for c in feature_cols if c in df.columns]
            if len(usable) >= 2:
                # PCA coordinates are recomputed from the same encoded feature columns
                # with the saved scaler and PCA artifacts.
                X = df[usable].replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.median()).fillna(0)
                X_scaled = artifacts["scaler"].transform(X)
                coords = artifacts["pca"].transform(X_scaled)
                plot_df = view.copy()
                plot_df["PCA 1"] = coords[df.index.get_indexer(view.index), 0]
                plot_df["PCA 2"] = coords[df.index.get_indexer(view.index), 1]
                fig, ax = plt.subplots(figsize=(9, 5))
                for segment in sorted(plot_df["Segment"].unique()):
                    sub = plot_df[plot_df["Segment"] == segment]
                    ax.scatter(sub["PCA 1"], sub["PCA 2"], s=22, alpha=0.65, label=segment)
                ax.set_xlabel("PCA component 1")
                ax.set_ylabel("PCA component 2")
                ax.legend(loc="best", fontsize=8)
                ax.grid(alpha=0.2)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Not enough saved feature columns to render the PCA chart.")
        except Exception as exc:
            st.info(f"PCA chart is unavailable: {exc}")

        st.markdown("### Segment summary")
        st.dataframe(get_profile(view), hide_index=True, use_container_width=True)

        # Customer lookup is also available directly on the Overview dashboard.
        st.markdown("---")
        st.subheader("🆕 Check a New Customer (Cold-Start Handling)")
        st.caption(
            "This lookup only checks existing dataset records and flags possible cold-start cases. "
            "It does not predict a segment for an unknown customer. Use Predict New Customer for that."
        )
        with st.form("overview_cold_start_check_form"):
            overview_cold_start_id = st.text_input(
                "Enter a customer ID to check",
                placeholder="Enter an existing Customer ID"
            ).strip()
            overview_check_cold_start = st.form_submit_button(
                "Check customer status", type="primary"
            )

        if overview_check_cold_start:
            if not overview_cold_start_id:
                st.warning("Enter a Customer ID first.")
            elif overview_cold_start_id not in df.index:
                st.warning(
                    f"Customer ID {overview_cold_start_id} is not in the current dataset. "
                    "Verify the ID or add the new customer's data before assigning a segment."
                )
                overview_action = get_new_customer_action("unknown_id")
                st.info(f"Suggested action: {overview_action['action']}")
            else:
                overview_status = classify_customer_status(overview_cold_start_id, df)
                if overview_status == "new_customer":
                    st.info(
                        "🆕 Cold-start customer — use an onboarding action instead of "
                        "treating the current segment as reliable."
                    )
                    overview_action = get_new_customer_action(overview_status)
                    st.markdown(f"**Profile:** {overview_action['description']}")
                    st.info(f"**Suggested action:** {overview_action['action']}")
                    overview_row = df.loc[overview_cold_start_id]
                    if isinstance(overview_row, pd.DataFrame):
                        overview_row = overview_row.iloc[0]
                    ca, cb, cc = st.columns(3)
                    ca.metric("Customer ID", overview_cold_start_id)
                    cb.metric(
                        "Customer tenure (days)",
                        int(overview_row.get("Customer_tenure_days", 0))
                    )
                    overview_total_purchases = sum(
                        float(overview_row.get(col, 0) or 0)
                        for col in [
                            "NumWebPurchases", "NumCatalogPurchases",
                            "NumStorePurchases", "NumDealsPurchases"
                        ]
                    )
                    cc.metric("Recorded purchases", int(overview_total_purchases))
                else:
                    st.success("Established customer — a trained cluster/segment is available.")
                    render_customer_details(overview_cold_start_id, df, artifacts)

        st.markdown("---")
        st.subheader("🔎 Look Up a Customer")
        st.write(
            "Search by ID or select a customer to view their cluster and full profile."
        )
        overview_ids = sorted(df.index.astype(str).unique().tolist())
        with st.form("overview_search_customer_form"):
            overview_customer_id = st.text_input(
                "Enter Customer ID",
                placeholder="For example, 5524"
            ).strip()
            overview_submitted = st.form_submit_button("Find customer", type="primary")

        if overview_submitted:
            if not overview_customer_id:
                st.warning("Enter a Customer ID first.")
            elif overview_customer_id not in df.index:
                st.error(
                    f"Customer ID {overview_customer_id} was not found. "
                    "Check the ID and try again."
                )
            else:
                render_customer_details(overview_customer_id, df, artifacts)

        overview_chosen_id = st.selectbox(
            "Or select an existing Customer ID",
            options=overview_ids,
            index=None,
            placeholder="Select customer ID...",
            key="overview_quick_customer_lookup"
        )
        if overview_chosen_id:
            render_customer_details(overview_chosen_id, df, artifacts)

    elif page == "Search Customer":
        # Cold-start check, matching the requested dashboard design.
        st.subheader("🆕 Check a New Customer (Cold-Start Handling)")
        st.caption(
            "This lookup only checks existing dataset records and flags possible cold-start cases. "
            "It does not predict a segment for an unknown customer. Use Predict New Customer for that."
        )
        with st.form("cold_start_check_form"):
            cold_start_id = st.text_input(
                "Enter a customer ID to check",
                placeholder="Enter an existing Customer ID"
            ).strip()
            check_cold_start = st.form_submit_button("Check customer status", type="primary")

        if check_cold_start:
            if not cold_start_id:
                st.warning("Enter a Customer ID first.")
            elif cold_start_id not in df.index:
                st.warning(
                    f"Customer ID {cold_start_id} is not in the current dataset. "
                    "Verify the ID or add the new customer's data before assigning a segment."
                )
                action = get_new_customer_action("unknown_id")
                st.info(f"Suggested action: {action['action']}")
            else:
                status = classify_customer_status(cold_start_id, df)
                if status == "new_customer":
                    st.info("🆕 Cold-start customer — use an onboarding action instead of treating the current segment as reliable.")
                    action = get_new_customer_action(status)
                    st.markdown(f"**Profile:** {action['description']}")
                    st.info(f"**Suggested action:** {action['action']}")
                    row = df.loc[cold_start_id]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    a, b, c = st.columns(3)
                    a.metric("Customer ID", cold_start_id)
                    b.metric("Customer tenure (days)", int(row.get("Customer_tenure_days", 0)))
                    total_purchases = sum(float(row.get(col, 0) or 0) for col in ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"])
                    c.metric("Recorded purchases", int(total_purchases))
                else:
                    st.success("Established customer — a trained cluster/segment is available.")
                    render_customer_details(cold_start_id, df, artifacts)

        st.markdown("---")
        st.subheader("🔎 Look Up a Customer")
        st.write("Search by ID or select a customer to view their cluster and full profile.")
        ids = sorted(df.index.astype(str).unique().tolist())
        with st.form("search_customer_form"):
            customer_id = st.text_input("Enter Customer ID", placeholder="For example, 5524").strip()
            submitted = st.form_submit_button("Find customer", type="primary")
        if submitted:
            if not customer_id:
                st.warning("Enter a Customer ID first.")
            elif customer_id not in df.index:
                st.error(f"Customer ID {customer_id} was not found. Check the ID and try again.")
            else:
                render_customer_details(customer_id, df, artifacts)
        chosen_id = st.selectbox(
            "Or select an existing Customer ID", options=ids, index=None,
            placeholder="Select customer ID...", key="quick_customer_lookup"
        )
        if chosen_id:
            render_customer_details(chosen_id, df, artifacts)

    elif page == "Explore Customers":
        st.subheader("Explore and filter customer records")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            available_clusters = sorted(df["Cluster"].dropna().unique().tolist())
            selected_clusters = st.multiselect(
                "Clusters to include", options=available_clusters,
                default=available_clusters,
                format_func=lambda value: f"Cluster {value} — {df.loc[df['Cluster'] == value, 'Segment'].iloc[0]}"
            )
        with c2:
            min_income = float(df["Income"].min()) if "Income" in df else 0.0
            max_income = float(df["Income"].max()) if "Income" in df else 1.0
            income_range = st.slider(
                "Income range", min_value=min_income, max_value=max_income,
                value=(min_income, max_income)
            ) if max_income > min_income else (min_income, max_income)
        with c3:
            max_spend = float(df["Total_Spending"].max()) if len(df) else 1.0
            spending_min = st.number_input("Minimum total spending", min_value=0.0, max_value=max(1.0, max_spend), value=0.0)
        with c4:
            id_query = st.text_input("Search ID contains", placeholder="Type part of an ID")
        filtered = df[df["Cluster"].isin(selected_clusters) & df["Segment"].isin(selected_segments)].copy()
        if "Income" in filtered:
            filtered = filtered[filtered["Income"].between(income_range[0], income_range[1])]
        filtered = filtered[filtered["Total_Spending"] >= spending_min]
        if id_query.strip():
            filtered = filtered[filtered.index.to_series().str.contains(id_query.strip(), case=False, na=False)]
        st.write(f"**{len(filtered):,} customer(s)** match these filters.")
        display_cols = [
            c for c in [
                "Cluster", "Segment", "Income", "Total_Spending", "Age",
                "Recency", "NumWebPurchases", "NumStorePurchases",
                "NumDealsPurchases", "Response"
            ] if c in filtered.columns
        ]
        st.dataframe(filtered[display_cols], use_container_width=True, height=420)

        st.markdown("#### Open a customer's full profile")
        if len(filtered):
            customer_options = filtered.index.astype(str).tolist()
            chosen_customer = st.selectbox(
                "Choose Customer ID", options=customer_options,
                format_func=lambda cid: f"{cid} — Cluster {int(filtered.loc[cid, 'Cluster'])} — {filtered.loc[cid, 'Segment']}"
            )
            if st.button("View customer details", type="primary"):
                render_customer_details(chosen_customer, df, artifacts)
        else:
            st.info("No customers match these filters. Choose another cluster or adjust the filters.")

        st.download_button(
            "Download filtered results",
            data=filtered.to_csv(index=True).encode("utf-8"),
            file_name="smartcart_filtered_customers.csv",
            mime="text/csv",
        )

    elif page == "Segment Profiles":
        st.subheader("Understand what makes each segment different")
        profile = get_profile(view)
        st.dataframe(profile, hide_index=True, use_container_width=True)
        numeric_choices = [
            c for c in ["Income", "Total_Spending", "Recency", "Age",
                        "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth"]
            if c in view.columns
        ]
        if numeric_choices:
            metric = st.selectbox("Compare segments by metric", numeric_choices, index=1 if "Total_Spending" in numeric_choices else 0)
            avg = view.groupby("Segment")[metric].mean().sort_values(ascending=False)
            st.bar_chart(avg, use_container_width=True)
        for segment in sorted(view["Segment"].unique()):
            with st.expander(f"📌 {segment}"):
                info = get_action(segment)
                st.write(f"**Profile:** {info.get('description', 'No description available.')}")
                st.write(f"**Suggested action:** {info.get('action', 'Review the segment before selecting a campaign.')}")

    elif page == "Model Evaluation":
        st.subheader("Model selection and quality")
        st.markdown(
            "The **elbow method** examines within-cluster sum of squares (WCSS/inertia); "
            "a bend in the curve suggests a useful cluster count. The **silhouette score** "
            "measures how well-separated the clusters are; values closer to 1 are generally better, "
            "values near 0 indicate overlap, and negative values may indicate poor assignments."
        )
        image_path = MODEL_DIR / "k_selection.png"
        c1, c2 = st.columns([1.3, 1])
        with c1:
            if image_path.exists():
                st.image(str(image_path), caption="Elbow method (WCSS) and silhouette score across candidate k values", use_container_width=True)
            else:
                st.warning("The saved elbow/silhouette chart is missing. Run `python train_model.py` to generate it.")
        with c2:
            comparison = artifacts["comparison"]
            st.metric("K-Means silhouette score", safe_float(comparison.get("kmeans_silhouette")))
            st.metric("Agglomerative silhouette score", safe_float(comparison.get("agglomerative_silhouette")))
            st.metric("K-Means Davies–Bouldin (lower is better)", safe_float(comparison.get("kmeans_davies_bouldin")))
            if "agglomerative_davies_bouldin" in comparison:
                st.metric("Agglomerative Davies–Bouldin", safe_float(comparison.get("agglomerative_davies_bouldin")))
            if "pca_explained_variance" in comparison:
                st.metric("PCA variance retained", f"{100*comparison['pca_explained_variance']:.1f}%")
        st.warning(
            "Interpretation notes: silhouette is not accuracy. In this run, k=2 has the highest silhouette "
            "and the geometric elbow estimate is shown in the candidate-k section; k=4 is retained for "
            "business-readable profiles, not because it wins both metrics. PCA variance retained is shown above. "
            "Campaign Response, marital status, education, age, and income are excluded from clustering inputs. "
            "SHAP explanations describe a Random Forest surrogate trained to mimic cluster labels; they do not "
            "explain K-Means directly."
        )
        st.info(
            "These scores are loaded from the model-training run. The elbow chart is generated by "
            "`train_model.py`; it evaluates candidate cluster counts before the final model is selected."
        )
        st.markdown("#### Candidate-k metrics")
        metrics_path = MODEL_DIR / "k_selection_metrics.csv"
        if metrics_path.exists():
            candidate_metrics = pd.read_csv(metrics_path)
            st.dataframe(candidate_metrics.round(3), hide_index=True, use_container_width=True)
            best_row = candidate_metrics.loc[candidate_metrics["silhouette"].idxmax()]
            st.warning(
                f"On this training run, silhouette is highest at k={int(best_row['k'])} "
                f"({best_row['silhouette']:.3f}). The dashboard retains k=4 to provide more actionable "
                "behavioral profiles, but this is a business/readability trade-off—not the silhouette optimum. "
                "Consider comparing k=2 and k=4 with campaign outcomes before production use."
            )
        st.markdown("#### Cluster counts in the saved model")
        st.bar_chart(df["Segment"].value_counts(), use_container_width=True)


    elif page == "DBSCAN Outlier Detection":
        st.subheader("DBSCAN — behavioral outlier detection")
        st.write(
            "K-Means remains the primary customer-segmentation algorithm. This DBSCAN tool is a "
            "complementary density-based detector that flags records in sparse regions as potential "
            "outliers. A DBSCAN noise label (-1) is not proof of fraud or bad data."
        )
        feature_cols = artifacts["feature_cols"]
        missing_features = [c for c in feature_cols if c not in df.columns]

        if missing_features:
            st.error("DBSCAN cannot run because these model features are missing: " + ", ".join(missing_features))
        elif df.empty:
            st.warning("There are no customer records available for DBSCAN.")
        else:
            left, right = st.columns(2)
            with left:
                eps = st.slider(
                    "Neighborhood radius (eps)",
                    min_value=0.1, max_value=5.0, value=0.8, step=0.1,
                    help="Distance threshold in the saved model's scaled feature space. Larger values usually reduce noise, but can merge clusters."
                )
            with right:
                min_samples = st.slider(
                    "Minimum samples", min_value=3, max_value=30, value=5, step=1,
                    help="Minimum number of nearby samples required for a core point, including the point itself."
                )

            compare_settings = st.checkbox(
                "Compare parameter settings",
                value=False,
                help="Runs a small parameter grid using the same saved scaler and feature set."
            )

            if st.button("Run DBSCAN outlier detection", type="primary"):
                try:
                    # Match the feature order used to train the saved scaler.
                    X_raw = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
                    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
                    if X_raw.isna().any().any():
                        X_raw = X_raw.fillna(X_raw.median()).fillna(0)

                    # Use the project's saved scaler, not a second scaler, so the eps slider
                    # remains in the same feature space used by the existing dashboard.
                    X_dbscan = artifacts["scaler"].transform(X_raw)
                    labels = DBSCAN(
                        eps=float(eps),
                        min_samples=int(min_samples),
                        n_jobs=-1
                    ).fit_predict(X_dbscan)

                    result = df.copy()
                    result["DBSCAN_Label"] = labels
                    result["Is_DBSCAN_Outlier"] = labels == -1
                    outliers = result[result["Is_DBSCAN_Outlier"]].copy()
                    clusters_found = len(set(labels) - {-1})
                    noise_count = int(np.sum(labels == -1))
                    noise_rate = 100 * noise_count / len(result) if len(result) else 0.0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Potential outliers", f"{noise_count:,}")
                    m2.metric("Outlier rate", f"{noise_rate:.1f}%")
                    m3.metric("Dense clusters found", f"{clusters_found:,}")

                    if noise_rate >= 40:
                        st.warning(
                            "A large share of records is currently labelled as noise. This is not automatically wrong. "
                            "Review the parameter comparison below, feature scaling, and whether the selected features are appropriate."
                        )
                    elif noise_rate == 0:
                        st.info(
                            "No noise points were detected. Check whether eps is so large that nearly all records belong to dense regions."
                        )

                    st.caption(
                        "DBSCAN results depend on eps, min_samples, feature scaling, and selected features. "
                        "Noise means 'not assigned to a dense cluster'; it does not mean fraud."
                    )

                    if compare_settings:
                        st.markdown("#### Parameter comparison")
                        grid_rows = []
                        # Use a compact grid around common values; evaluate in the exact same scaled feature space.
                        eps_candidates = sorted(set([0.4, 0.6, 0.8, 1.0, 1.2, 1.5, float(eps)]))
                        min_samples_candidates = sorted(set([5, 8, 10, int(min_samples)]))
                        for test_eps in eps_candidates:
                            for test_min_samples in min_samples_candidates:
                                test_labels = DBSCAN(
                                    eps=float(test_eps),
                                    min_samples=int(test_min_samples),
                                    n_jobs=-1
                                ).fit_predict(X_dbscan)
                                test_noise = int(np.sum(test_labels == -1))
                                test_clusters = len(set(test_labels) - {-1})
                                grid_rows.append({
                                    "eps": test_eps,
                                    "min_samples": test_min_samples,
                                    "dense_clusters": test_clusters,
                                    "outliers": test_noise,
                                    "outlier_rate_%": round(100 * test_noise / len(test_labels), 2)
                                })
                        grid_df = pd.DataFrame(grid_rows).sort_values(
                            ["outlier_rate_%", "dense_clusters"], ascending=[True, False]
                        )
                        st.dataframe(grid_df, hide_index=True, use_container_width=True)
                        st.caption(
                            "Do not select settings only because they produce fewer outliers. "
                            "Choose settings after checking cluster sizes, business interpretability, and stability."
                        )

                    st.markdown("#### Outlier records")
                    if outliers.empty:
                        st.info("No noise points were detected with these settings. Try adjusting eps or min_samples if appropriate.")
                    else:
                        display_cols = [
                            c for c in [
                                "Cluster", "Segment", "Total_Spending", "Total_Purchases",
                                "Recency", "NumWebVisitsMonth", "DBSCAN_Label"
                            ] if c in outliers.columns
                        ]
                        st.dataframe(
                            outliers[display_cols].reset_index(),
                            hide_index=True,
                            use_container_width=True
                        )

                    st.download_button(
                        "Download DBSCAN results (CSV)",
                        data=result.to_csv(index=True).encode("utf-8"),
                        file_name="smartcart_dbscan_outlier_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"DBSCAN could not run: {exc}")
            else:
                st.info(
                    "Choose parameters, then click **Run DBSCAN outlier detection** to generate labels "
                    "and review potential outliers. Enable parameter comparison to explore alternatives."
                )


    elif page == "Predict New Customer":
        st.subheader("🧭 Assign a segment to a new customer profile")
        st.write(
            "Enter observed behavioral values for a customer not present in the dataset. "
            "The app applies the saved scaler, PCA transformation, and K-Means model. "
            "This is an unsupervised segment assignment—not a supervised purchase prediction."
        )
        with st.form("new_customer_prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                recency = st.number_input("Days since last purchase (Recency)", min_value=0, max_value=3650, value=30)
                total_spending = st.number_input("Total product spending", min_value=0.0, value=500.0, step=50.0)
                web_purchases = st.number_input("Web purchases", min_value=0, max_value=1000, value=3)
                catalog_purchases = st.number_input("Catalog purchases", min_value=0, max_value=1000, value=2)
            with c2:
                store_purchases = st.number_input("Store purchases", min_value=0, max_value=1000, value=4)
                deal_purchases = st.number_input("Deal purchases", min_value=0, max_value=1000, value=1)
                web_visits = st.number_input("Web visits in a month", min_value=0, max_value=1000, value=5)
            with c3:
                st.markdown("**Optional context**")
                st.caption("Only behavioral fields below affect the cluster assignment.")
                st.caption("Income, age, education, marital status and campaign response are not model inputs.")
                no_history = st.checkbox("This is a genuinely new customer with no purchase history yet")
            predict_submitted = st.form_submit_button("Assign customer segment", type="primary")

        if predict_submitted:
            total_purchases = web_purchases + catalog_purchases + store_purchases + deal_purchases
            channel_total = web_purchases + catalog_purchases + store_purchases
            new_profile = {
                "Recency": float(recency),
                "Total_Spending": float(total_spending),
                "Total_Purchases": float(total_purchases),
                "NumWebPurchases": float(web_purchases),
                "NumCatalogPurchases": float(catalog_purchases),
                "NumStorePurchases": float(store_purchases),
                "NumDealsPurchases": float(deal_purchases),
                "NumWebVisitsMonth": float(web_visits),
                "Online_Purchase_Share": float(web_purchases / channel_total) if channel_total else 0.0,
                "Deal_Purchase_Share": float(deal_purchases / total_purchases) if total_purchases else 0.0,
                "Web_Visit_to_Purchase_Ratio": float(web_visits / (web_purchases + 1)),
            }
            try:
                # A truly historyless customer receives an onboarding state, not a fabricated K-Means persona.
                if no_history or (total_purchases == 0 and total_spending == 0):
                    predicted_cluster = None
                    predicted_segment = "New Customer Onboarding (Provisional)"
                    assignment_type = "cold_start_onboarding"
                    st.info("No purchase history is available. This is a provisional onboarding status, not a K-Means prediction.")
                    action = get_new_customer_action("new_customer")
                else:
                    feature_cols = artifacts["feature_cols"]
                    profile_df = pd.DataFrame([{c: new_profile[c] for c in feature_cols}], columns=feature_cols)
                    scaled = artifacts["scaler"].transform(profile_df)
                    projected = artifacts["pca"].transform(scaled)
                    predicted_cluster = int(artifacts["kmeans"].predict(projected)[0])
                    predicted_segment = artifacts["cluster_names"].get(predicted_cluster, f"Cluster {predicted_cluster}")
                    assignment_type = "behavior_profile_cluster_assignment"
                    action = get_action(predicted_segment)
                    st.warning("This is an assignment to an existing cluster based on supplied behavior, not a prediction of future purchases.")

                # Persist the submitted profile and result so the result survives app restarts.
                save_profile_assignment(
                    profile=new_profile,
                    segment=predicted_segment,
                    cluster_id=predicted_cluster,
                    assignment_type=assignment_type,
                )
                st.success(f"Result: **{predicted_segment}**" + (f" (Cluster {predicted_cluster})" if predicted_cluster is not None else ""))
                a, b, c = st.columns(3)
                a.metric("Total spending supplied", f"{total_spending:,.0f}")
                b.metric("Purchases supplied", f"{total_purchases:,}")
                c.metric("Days since last purchase", f"{recency:,}")
                st.markdown("#### Suggested next action")
                st.write(action.get("description", "Review the profile before selecting an action."))
                st.info(action.get("action", "Review manually."))
                st.caption("As new transactions arrive, update the customer's behavioral profile and reassign the cluster. The provisional onboarding state should not be presented as a learned segment.")
            except Exception as exc:
                st.error(f"Could not assign or save the profile: {exc}")

    elif page == "Assignment History":
        st.subheader("Saved new-customer assignments")
        st.caption("These entries are stored in smartcart.db and remain available after restarting the local app.")
        history = list_profile_assignments(limit=500)
        if not history:
            st.info("No new-customer profiles have been assigned yet.")
        else:
            hist_df = pd.DataFrame([{k: row[k] for k in ["id", "customer_id", "assignment_type", "cluster_id", "segment", "created_at"]} for row in history])
            st.dataframe(hist_df, hide_index=True, use_container_width=True)
            selected_assignment = st.selectbox("Inspect saved profile", options=hist_df["id"].tolist(), format_func=lambda x: f"#{x} — {hist_df.loc[hist_df['id'] == x, 'segment'].iloc[0]}")
            selected = next((row for row in history if row["id"] == selected_assignment), None)
            if selected:
                st.json(selected["profile"])
            st.download_button("Download assignment history", data=hist_df.to_csv(index=False).encode("utf-8"), file_name="smartcart_assignment_history.csv", mime="text/csv")


if __name__ == "__main__":
    main()
