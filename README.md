# SmartCart Customer Intelligence

SmartCart is a Streamlit customer-segmentation prototype using behavior-first features, K-Means, model evaluation, customer lookup, and new-profile assignment.

## Run locally (Windows PowerShell)

1. Open this `smartcart_deploy` folder in VS Code.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Configure login credentials. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and replace both example values with your own unique credentials. Do not commit `secrets.toml` to GitHub.
4. Start the app:

```powershell
python -m streamlit run app.py
```

5. Open the local URL Streamlit prints, usually `http://localhost:8501`, and sign in using your configured credentials.

If credentials are missing or invalid, the app denies access; there is no demo-access bypass. For Streamlit Community Cloud, configure the same `[auth]` values in the app's Secrets settings instead of uploading `secrets.toml`.

## Dashboard pages
- **Overview** — customer counts, segment distribution, PCA map, profile summary, existing-record cold-start heuristic, and customer lookup.
- **Search Customer** — look up an existing customer ID and inspect their profile.
- **Explore Customers** — filter customers and export filtered rows.
- **Segment Profiles** — compare segment-level behavior and proposed campaign experiments.
- **Model Evaluation** — elbow/WCSS, candidate-k silhouette table, K-Means vs Agglomerative comparison, Davies–Bouldin, and PCA variance retained.
- **Predict New Customer** — assign a supplied behavioral profile to a trained cluster, or explicitly mark a genuinely historyless customer to receive a **provisional onboarding state** instead of a fabricated K-Means label.
- **Assignment History** — review and export new-profile assignments saved in SQLite.

## Database persistence

On first launch, `database.py` creates `smartcart.db` and imports the bundled `model/segmented_customers.csv` records once. The dashboard reads existing customer records from SQLite, and new-profile assignments are saved to `profile_assignments` and remain after app restarts. The database file is excluded from Git by `.gitignore`.

**Deployment boundary:** this SQLite setup provides persistence for a local/demo app, not production-grade multi-instance persistence. For a public multi-user deployment, use PostgreSQL (or another managed database), configure its connection through deployment secrets, add backups/migrations, and apply access controls. Do not expose customer data publicly.

## Modeling decisions and limitations
- Clustering inputs are behavioral: recency, total spending, purchase frequency, channel counts, deal purchases, web visits, and derived channel/deal ratios.
- Marital status, education, age, income, and campaign `Response` are not clustering inputs. Some fields may remain visible in the customer profile for analysis only.
- Four clusters are retained for business-readable segment profiles. On the current run, silhouette is higher at k=2 than k=4; k=4 is an interpretability trade-off, not the silhouette optimum.
- Segment names are rule-based summaries of cluster behavior, not validated identities or proven personas.
- The existing-ID cold-start check only checks records already in the dataset. A historyless new customer receives a provisional onboarding state. When behavioral data is available, the app can assign it to a trained cluster; this is not a supervised prediction of future buying behavior.
- SHAP uses a Random Forest surrogate trained to mimic cluster labels. It does not explain K-Means internals and should not be interpreted causally.
- Marketing recommendations are hypotheses for testing, not guaranteed outcomes. Evaluate campaigns with holdout groups and business metrics before deployment.

## Files
- `app.py` — Streamlit dashboard.
- `auth.py` — fail-closed credential gate.
- `database.py` — SQLite initialization, customer import, and saved assignment history.
- `preprocessing.py` — behavior feature engineering and scaling.
- `train_model.py` — PCA, K-Means, Agglomerative baseline, metrics, personas, artifacts.
- `explainability.py` — surrogate model explanation helper.
- `cold_start.py` — existing-record cold-start heuristic.
- `marketing_actions.py` — segment action suggestions.
- `data/smartcart_customers.csv` — source customer dataset.
- `model/` — generated model artifacts, segmented customer file, and evaluation metrics.

Run `python train_model.py` to retrain and regenerate model artifacts.


## DBSCAN outlier detection
The dashboard includes a **DBSCAN Outlier Detection** page. K-Means remains the primary segmentation algorithm; DBSCAN is an additional density-based detector for potential outliers. The page lets you tune `eps` and `min_samples`, reports noise points labelled `-1`, and exports all records with DBSCAN labels to CSV. Noise points should be reviewed, not automatically treated as errors or fraud.
