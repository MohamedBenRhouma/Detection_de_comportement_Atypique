import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create output directory
OUT_DIR = "PFE_Visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_excel("results_with_anomalies.xlsx", sheet_name="All Transactions", engine="openpyxl")

# Find numeric columns that were used for modeling
_exclude = ["anomaly_IF", "anomaly_score_IF", "anomaly_Distance", "mahalanobis_distance", 
            "status_IF", "status_Distance", "high_confidence_fraud", "CHRONO"]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
viz_cols = [c for c in numeric_cols if c not in _exclude and not c.endswith("_enc")] 
# Note: we use all numeric features for PCA. Actually, let's use the exact 28 if possible.
# If we just take all numeric minus excludes, it's fine for PCA.

print(f"Computing 3D PCA on {len(viz_cols)} features...")
X = df[viz_cols].fillna(0).values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

var_exp = pca.explained_variance_ratio_ * 100

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
df_pca["anomaly_IF"] = df["anomaly_IF"].map({1: "Normal", -1: "Anomaly"})
df_pca["mahalanobis_distance"] = df["mahalanobis_distance"]
df_pca["high_confidence_fraud"] = df["high_confidence_fraud"]

# =============================================================================
# 1. 3D ISOLATION FOREST VISUALIZATION
# =============================================================================
print("Generating Isolation Forest 3D Plot...")
fig_if = px.scatter_3d(
    df_pca, x="PC1", y="PC2", z="PC3",
    color="anomaly_IF",
    color_discrete_map={"Normal": "#3fb950", "Anomaly": "#f85149"},
    opacity=0.7,
    title="<b>Isolation Forest: 3D Structural Anomaly Detection</b><br><sup>Les points rouges (Anomalies) sont structurellement isolés du noyau dense (Vert).</sup>"
)

fig_if.update_traces(marker=dict(size=3, line=dict(width=0)))
fig_if.update_layout(
    scene=dict(
        xaxis_title=f"Composante Principale 1 ({var_exp[0]:.1f}%)",
        yaxis_title=f"Composante Principale 2 ({var_exp[1]:.1f}%)",
        zaxis_title=f"Composante Principale 3 ({var_exp[2]:.1f}%)",
        bgcolor="rgba(0,0,0,0)"
    ),
    paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9"),
    margin=dict(l=0, r=0, b=0, t=50)
)
fig_if.write_html(f"{OUT_DIR}/01_Isolation_Forest_3D.html")


# =============================================================================
# 2. 3D MAHALANOBIS DISTANCE VISUALIZATION (Gradient)
# =============================================================================
print("Generating Mahalanobis 3D Plot...")
fig_mah = go.Figure()

# Plot normal
fig_mah.add_trace(go.Scatter3d(
    x=df_pca["PC1"], y=df_pca["PC2"], z=df_pca["PC3"],
    mode="markers",
    marker=dict(
        size=3,
        color=df_pca["mahalanobis_distance"],
        colorscale="Viridis", # Dark purple to bright yellow
        opacity=0.8,
        colorbar=dict(title="Distance<br>Statistique"),
        showscale=True
    ),
    text=df_pca["mahalanobis_distance"],
    hovertemplate="Distance: %{text:.2f}<extra></extra>"
))

fig_mah.update_layout(
    title="<b>Mahalanobis Distance: 3D Statistical Distance</b><br><sup>Les points jaunes s'éloignent statistiquement du centre de gravité des données.</sup>",
    scene=dict(
        xaxis_title=f"PC1 ({var_exp[0]:.1f}%)",
        yaxis_title=f"PC2 ({var_exp[1]:.1f}%)",
        zaxis_title=f"PC3 ({var_exp[2]:.1f}%)",
        bgcolor="rgba(0,0,0,0)"
    ),
    paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9"),
    margin=dict(l=0, r=0, b=0, t=50)
)
fig_mah.write_html(f"{OUT_DIR}/02_Mahalanobis_Distance_3D.html")

# =============================================================================
# 3. HIGH CONFIDENCE FRAUD (Both Models)
# =============================================================================
print("Generating Intersection 3D Plot...")
df_pca["Intersection"] = "Normal (ou un seul modèle)"
df_pca.loc[df_pca["high_confidence_fraud"] == 1, "Intersection"] = "Anomalie Haute Confiance (Les 2 Modèles)"

fig_int = px.scatter_3d(
    df_pca, x="PC1", y="PC2", z="PC3",
    color="Intersection",
    color_discrete_map={"Normal (ou un seul modèle)": "#8b949e", "Anomalie Haute Confiance (Les 2 Modèles)": "#ff0000"},
    opacity=0.6,
    title="<b>Intersection des Modèles (Tableau Croisé)</b><br><sup>Transactions signalées simultanément par les DEUX modèles (Extrême urgence).</sup>"
)
fig_int.update_traces(marker=dict(size=3.5, line=dict(width=0)))
fig_int.update_layout(
    scene=dict(bgcolor="rgba(0,0,0,0)"),
    paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9"),
    margin=dict(l=0, r=0, b=0, t=50)
)
fig_int.write_html(f"{OUT_DIR}/03_Intersection_Models_3D.html")

print(f"\n[SUCCESS] Generated 3 beautiful 3D interactive plots in '{OUT_DIR}' folder.")
print("Open these .html files in your browser. You can rotate them and take perfect screenshots for your PFE Report!")
