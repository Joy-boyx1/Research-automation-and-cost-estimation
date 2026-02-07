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

if dfs:
    selected_file = st.selectbox("📂 Sélectionnez un fichier :", list(dfs.keys()))
    st.dataframe(dfs[selected_file])

random_title = st.text_input("🔍 Entrez un titre ou mot-clé à rechercher :")
keyword_search = st.checkbox("🔑 Recherche par mot-clé exact (100 % similaire)")

if random_title and dfs:

    results_rows = []

    for name, df in dfs.items():
        intitules = df.iloc[:, 1].dropna().astype(str).tolist()  # colonne B

        if keyword_search:
            for idx, text in enumerate(intitules):
                if random_title.upper() in text.upper():
                    results_rows.append({
                        "Fichier": name,
                        "Intitulé affaire": df.iloc[idx, 1],
                        "Montant Budgetisé": df.iloc[idx, 9],
                        "Estimation financière": df.iloc[idx, 10]
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
                    "Estimation financière": df.iloc[idx, 10]
                })

    if results_rows:
        # Stocker dans session_state pour suppression
        st.session_state.results_df = pd.DataFrame(results_rows)

        st.subheader("📊 Affaires trouvées (cliquez sur 🗑️ pour supprimer une ligne)")

        df_display = st.session_state.results_df
        to_delete = None  # variable pour stocker l'index à supprimer

        # Affichage ligne par ligne avec bouton corbeille
        for i in range(len(df_display)):
            row = df_display.iloc[i]
            row_text = f"{row['Intitulé affaire']} | {row['Montant Budgetisé']} | {row['Estimation financière']} | {row['Fichier']}"
            if st.button(f"🗑️ Supprimer : {row_text}", key=f"del_{i}"):
                to_delete = df_display.index[i]

        if to_delete is not None:
            st.session_state.results_df = st.session_state.results_df.drop(to_delete).reset_index(drop=True)
            st.experimental_rerun()

    else:
        st.warning("⚠️ Aucun résultat trouvé.")
