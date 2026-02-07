import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("📊 Recherche Automatisée et Clustering des Plannings")

# Liste des fichiers attendus
expected_files = [f"Consultation du planning des af {year}.xlsx" for year in range(2015, 2025)]

# Chargement du modèle SentenceTransformer
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

model = load_model()

# Upload des fichiers Excel
uploaded_files = st.file_uploader(
    "📂 Importez vos fichiers Excel",
    type=["xlsx"],
    accept_multiple_files=True
)

dfs = {}
if uploaded_files:
    for file in uploaded_files:
        if file.name in expected_files:
            try:
                df = pd.read_excel(file, header=0, engine='openpyxl')
                dfs[file.name] = df
                st.success(f"✅ {file.name} chargé avec succès !")
            except Exception as e:
                st.error(f"❌ Erreur de lecture du fichier {file.name} : {e}")
        else:
            st.warning(f"⚠️ Fichier ignoré : {file.name} (Nom non reconnu)")

# Affichage d’un fichier pour vérification
if dfs:
    selected_file = st.selectbox("📂 Sélectionnez un fichier :", list(dfs.keys()))
    st.dataframe(dfs[selected_file])

# Input utilisateur
random_title = st.text_input("🔍 Entrez un titre ou mot-clé à rechercher :")
keyword_search = st.checkbox("🔑 Recherche par mot-clé exact (100 % similaire)")

# ===============================
# RECHERCHE
# ===============================
if random_title and dfs:

    results_rows = []

    for name, df in dfs.items():
        intitules = df.iloc[:, 1].dropna().astype(str).tolist()  # colonne B
        sites = df.iloc[:, 3].astype(str).tolist()  # colonne D

        if keyword_search:
            for idx, text in enumerate(intitules):
                if random_title.upper() in text.upper():
                    results_rows.append({
                        "Fichier": name,
                        "Intitulé affaire": df.iloc[idx, 1],
                        "Montant Budgetisé": df.iloc[idx, 9],
                        "Estimation financière": df.iloc[idx, 10],
                        "Site": df.iloc[idx, 3]
                    })
        else:
            query_embedding = model.encode([random_title])
            embeddings_other = model.encode(intitules)
            similarity_matrix = cosine_similarity(query_embedding, embeddings_other)
            high_sim_indices = np.where(similarity_matrix[0] > 0.7)[0]

            for idx in high_sim_indices:
                results_rows.append({
                    "Fichier": name,
                    "Intitulé affaire": df.iloc[idx, 1],
                    "Montant Budgetisé": df.iloc[idx, 9],
                    "Estimation financière": df.iloc[idx, 10],
                    "Site": df.iloc[idx, 3]
                })

    if results_rows:
        st.session_state.results_df = pd.DataFrame(results_rows)

        # Filtrer par Site
        sites_dispo = st.session_state.results_df["Site"].unique().tolist()
        site_filter = st.multiselect("Filtrer par Site :", options=sites_dispo, default=sites_dispo)
        df_filtered = st.session_state.results_df[st.session_state.results_df["Site"].isin(site_filter)]

        st.subheader("📊 Affaires trouvées (cochez pour supprimer)")
        df_display = df_filtered.copy()
        to_delete_indices = []

        # Affichage ligne par ligne avec checkbox
        for i in range(len(df_display)):
            row = df_display.iloc[i]
            checked = st.checkbox(
                f"{row['Intitulé affaire']} | {row['Montant Budgetisé']} | {row['Estimation financière']} | {row['Site']} | {row['Fichier']}",
                key=f"chk_{i}"
            )
            if checked:
                to_delete_indices.append(df_display.index[i])

        # Bouton global pour supprimer toutes les lignes cochées
        if st.button("🗑️ Supprimer la sélection"):
            if to_delete_indices:
                st.session_state.results_df.drop(index=to_delete_indices, inplace=True)
                st.session_state.results_df.reset_index(drop=True, inplace=True)
                st.success("✅ Lignes supprimées avec succès")
            else:
                st.warning("⚠️ Aucune ligne cochée à supprimer")

        # Affichage final du tableau filtré
        st.dataframe(st.session_state.results_df[st.session_state.results_df["Site"].isin(site_filter)], use_container_width=True)

        # ===============================
        # STATISTIQUES
        # ===============================
        df_stats = st.session_state.results_df[st.session_state.results_df["Site"].isin(site_filter)].copy()
        montant_nonzero = df_stats[df_stats["Montant Budgetisé"] != 0]["Montant Budgetisé"]
        estimation_nonzero = df_stats[df_stats["Estimation financière"] != 0]["Estimation financière"]

        st.subheader("📊 Statistiques")
        if len(montant_nonzero) > 0:
            st.write("**Montant Budgetisé**")
            st.write(f"Moyenne : {montant_nonzero.mean():.2f}")
            st.write(f"Médiane : {montant_nonzero.median():.2f}")
            st.write(f"Ecart-type : {montant_nonzero.std():.2f}")
        if len(estimation_nonzero) > 0:
            st.write("**Estimation financière**")
            st.write(f"Moyenne : {estimation_nonzero.mean():.2f}")
            st.write(f"Médiane : {estimation_nonzero.median():.2f}")
            st.write(f"Ecart-type : {estimation_nonzero.std():.2f}")

        # Moyenne combinée
        if len(montant_nonzero) > 0 and len(estimation_nonzero) > 0:
            moyenne_combinee = (montant_nonzero.mean() + estimation_nonzero.mean()) / 2
        elif len(montant_nonzero) > 0:
            moyenne_combinee = montant_nonzero.mean()
        elif len(estimation_nonzero) > 0:
            moyenne_combinee = estimation_nonzero.mean()
        else:
            moyenne_combinee = None
        if moyenne_combinee is not None:
            st.write(f"**Moyenne combinée : {moyenne_combinee:.2f}**")

        # ===============================
        # CLUSTERING AUTOMATIQUE
        # ===============================
        st.subheader("🤖 Clustering automatique")

        df_ml = df_stats.copy()
        df_ml = df_ml[(df_ml["Montant Budgetisé"] != 0) | (df_ml["Estimation financière"] != 0)]
        if not df_ml.empty:
            text_embeddings = model.encode(df_ml["Intitulé affaire"].tolist())
            numeric_data = df_ml[["Montant Budgetisé", "Estimation financière"]].fillna(0).values
            numeric_scaled = StandardScaler().fit_transform(numeric_data)
            features = np.hstack([text_embeddings, numeric_scaled])

            n_clusters = st.slider("Nombre de clusters :", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_ml['Cluster'] = kmeans.fit_predict(features)

            st.dataframe(df_ml[["Intitulé affaire","Montant Budgetisé","Estimation financière","Site","Cluster"]])

            # ----------------------------
            # Diagramme interactif du clustering
            # ----------------------------
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(features)
            df_ml['PCA1'] = reduced_features[:,0]
            df_ml['PCA2'] = reduced_features[:,1]

            fig = px.scatter(
                df_ml,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                hover_data=['Intitulé affaire','Montant Budgetisé','Estimation financière','Site'],
                title="Diagramme du clustering (PCA 2D)"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Aucun résultat trouvé.")
