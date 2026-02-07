import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl

st.title("📊 Recherche Automatisée dans l'historique des Plannings")

expected_files = [
    f"Consultation du planning des af {year}.xlsx" for year in range(2015, 2025)
]

# 🔁 MODÈLE XLM-ROBERTA MULTILINGUE
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

model = load_model()

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

# affichage brut d’un fichier
if dfs:
    selected_file = st.selectbox("📂 Sélectionnez un fichier à afficher :", list(dfs.keys()))
    st.dataframe(dfs[selected_file])

random_title = st.text_input("🔍 Entrez un titre à comparer :")

# ===============================
# RECHERCHE DE SIMILARITÉ + TABLEAU STRUCTURÉ
# ===============================
if random_title and dfs:
    st.write("📊 Calcul des similarités en cours...")

    random_title_embedding = model.encode([random_title])
    results_rows = []

    for name, df in dfs.items():
        if df.shape[1] > 1:
            intitule_other = df.iloc[:, 1].dropna().astype(str).tolist()

            if intitule_other:
                embeddings_other = model.encode(intitule_other)
                similarity_matrix = cosine_similarity(random_title_embedding, embeddings_other)

                high_sim_indices = np.where(similarity_matrix[0] > 0.7)[0]

                for idx in high_sim_indices:
                    intitule = df.iloc[idx, 1]

                    # adapter index si besoin
                    montant_budget = df.iloc[idx, 5] if df.shape[1] > 5 else "N/A"
                    estimation = df.iloc[idx, 6] if df.shape[1] > 6 else "N/A"

                    results_rows.append({
                        "Fichier": name,
                        "Intitulé affaire": intitule,
                        "Montant Budgetisé": montant_budget,
                        "Estimation financière": estimation
                    })

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        st.subheader("📊 Affaires similaires trouvées")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("⚠️ Aucune similarité supérieure à 0.7 trouvée.")
