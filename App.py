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

# 🔁 Chargement du modèle XLM-R / MPNet multilingue
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
            # 🔹 Recherche mot-clé exact
            for idx, text in enumerate(intitules):
                if random_title.upper() in text.upper():  # ignore casse
                    results_rows.append({
                        "Fichier": name,
                        "Intitulé affaire": df.iloc[idx, 1],   # colonne B
                        "Montant Budgetisé": df.iloc[idx, 9],  # colonne J
                        "Estimation financière": df.iloc[idx, 10]  # colonne K
                    })
        else:
            # 🔹 Recherche embeddings classique
            query_embedding = model.encode([random_title])
            embeddings_other = model.encode(intitules)
            similarity_matrix = cosine_similarity(query_embedding, embeddings_other)
            high_sim_indices = np.where(similarity_matrix[0] > 0.7)[0]

            for idx in high_sim_indices:
                results_rows.append({
                    "Fichier": name,
                    "Intitulé affaire": df.iloc[idx, 1],   # colonne B
                    "Montant Budgetisé": df.iloc[idx, 9],  # colonne J
                    "Estimation financière": df.iloc[idx, 10]  # colonne K
                })

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        st.subheader("📊 Affaires trouvées")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("⚠️ Aucun résultat trouvé.")
