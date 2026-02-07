import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl

st.title("📊 Recherche Automatisée dans l'historique des Plannings")

# Liste des fichiers attendus
expected_files = [
    f"Consultation du planning des af {year}.xlsx" for year in range(2015, 2025)
]

# Chargement du modèle
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

        if keyword_search:
            # Recherche mot-clé exact
            for idx, text in enumerate(intitules):
                if random_title.upper() in text.upper():
                    results_rows.append({
                        "Fichier": name,
                        "Intitulé affaire": df.iloc[idx, 1],
                        "Montant Budgetisé": df.iloc[idx, 9],
                        "Estimation financière": df.iloc[idx, 10]
                    })
        else:
            # Recherche embeddings
            query_embedding = model.encode([random_title])
            embeddings_other = model.encode(intitules)
            similarity_matrix = cosine_similarity(query_embedding, embeddings_other)
            high_sim_indices = np.where(similarity_matrix[0] > 0.9)[0]

            for idx in high_sim_indices:
                results_rows.append({
                    "Fichier": name,
                    "Intitulé affaire": df.iloc[idx, 1],
                    "Montant Budgetisé": df.iloc[idx, 9],
                    "Estimation financière": df.iloc[idx, 10]
                })

    if results_rows:
        # Stocker dans session_state pour suppression
        if 'results_df' not in st.session_state:
            st.session_state.results_df = pd.DataFrame(results_rows)
        else:
            st.session_state.results_df = pd.DataFrame(results_rows)

        st.subheader("📊 Affaires trouvées (cochez pour supprimer)")

        df_display = st.session_state.results_df.copy()

        # Ajouter une colonne checkbox
        df_display['Supprimer'] = False
        for i in range(len(df_display)):
            df_display.at[i, 'Supprimer'] = st.checkbox(
                f"{df_display.iloc[i]['Intitulé affaire']} | {df_display.iloc[i]['Montant Budgetisé']} | {df_display.iloc[i]['Estimation financière']} | {df_display.iloc[i]['Fichier']}",
                key=f"chk_{i}"
            )

        # Bouton global pour supprimer toutes les lignes cochées
        if st.button("🗑️ Supprimer la sélection"):
            # Supprimer toutes les lignes où 'Supprimer' est True
            st.session_state.results_df = st.session_state.results_df[
                [not st.session_state.results_df.index[i] in df_display[df_display['Supprimer']].index for i in range(len(df_display))]
            ].reset_index(drop=True)
            st.success("✅ Lignes supprimées")
            st.experimental_rerun()

        # Affichage final du tableau
        st.dataframe(st.session_state.results_df, use_container_width=True)

    else:
        st.warning("⚠️ Aucun résultat trouvé.")
