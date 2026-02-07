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

# 🔁 NOUVEAU MODÈLE XLM-ROBERTA MULTILINGUE
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
    selected_file = st.selectbox("📂 Sélectionnez un fichier à afficher :", list(dfs.keys()))
    st.dataframe(dfs[selected_file])

random_title = st.text_input("🔍 Entrez un titre à comparer :")

if random_title and dfs:
    st.write("📊 Calcul des similarités en cours...")

    random_title_embedding = model.encode([random_title])

    similarity_results = {}

    for name, df in dfs.items():
        if df.shape[1] > 1:
            intitule_other = df.iloc[:, 1].dropna().astype(str).tolist()

            if intitule_other:
                embeddings_other = model.encode(intitule_other)
                similarity_matrix = cosine_similarity(random_title_embedding, embeddings_other)
                similarity_results[name] = similarity_matrix

    found_similarities = False

    for name, similarity_matrix in similarity_results.items():
        high_sim_indices = np.where(similarity_matrix[0] > 0.7)[0]

        if len(high_sim_indices) > 0:
            st.subheader(f"📁 Résultats pour {name}")

            for idx in high_sim_indices:
                similarity_score = similarity_matrix[0, idx]
                matching_sentence = dfs[name].iloc[idx, 1]
                site_value = dfs[name].iloc[idx, 3] if dfs[name].shape[1] > 3 else "N/A"

                st.write(f"🔹 **Similarité :** {similarity_score:.4f}")
                st.write(f"📌 **Phrase correspondante :** {matching_sentence}")
                st.write(f"📍 **Site :** {site_value}")
                st.markdown("---")

            found_similarities = True

    if not found_similarities:
        st.warning("⚠️ Aucune similarité supérieure à 0.7 trouvée.")
