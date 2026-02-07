import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Recherche Automatisée dans l'historique des Plannings")

# Liste des fichiers attendus
expected_files = [f"Consultation du planning des af {year}.xlsx" for year in range(2015, 2025)]

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
        st.session_state.results_df = pd.DataFrame(results_rows)
        st.subheader("📊 Affaires trouvées (cochez pour supprimer)")

        df_display = st.session_state.results_df.copy()
        to_delete_indices = []

        # Affichage ligne par ligne avec checkbox
        for i in range(len(df_display)):
            row = df_display.iloc[i]
            checked = st.checkbox(
                f"{row['Intitulé affaire']} | {row['Montant Budgetisé']} | {row['Estimation financière']} | {row['Fichier']}",
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

        # Affichage final du tableau
        st.dataframe(st.session_state.results_df, use_container_width=True)

        # ===============================
        # STATISTIQUES ET DISTRIBUTIONS
        # ===============================
        df_stats = st.session_state.results_df.copy()
        montant_nonzero = df_stats[df_stats["Montant Budgetisé"] != 0]["Montant Budgetisé"]
        estimation_nonzero = df_stats[df_stats["Estimation financière"] != 0]["Estimation financière"]

        st.subheader("📊 Statistiques")

        # Montant Budgetisé
        if len(montant_nonzero) > 0:
            st.write("**Montant Budgetisé**")
            st.write(f"Moyenne : {montant_nonzero.mean():.2f}")
            st.write(f"Médiane : {montant_nonzero.median():.2f}")
            st.write(f"Ecart-type : {montant_nonzero.std():.2f}")
        else:
            st.warning("⚠️ La colonne Montant Budgetisé contient uniquement des 0")

        # Estimation financière
        if len(estimation_nonzero) > 0:
            st.write("**Estimation financière**")
            st.write(f"Moyenne : {estimation_nonzero.mean():.2f}")
            st.write(f"Médiane : {estimation_nonzero.median():.2f}")
            st.write(f"Ecart-type : {estimation_nonzero.std():.2f}")
        else:
            st.warning("⚠️ La colonne Estimation financière contient uniquement des 0")

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
        # HISTOGRAMMES
        # ===============================
        st.subheader("📊 Histogrammes")

        if len(montant_nonzero) > 0:
            plt.figure(figsize=(8, 4))
            plt.bar(df_stats["Intitulé affaire"], df_stats["Montant Budgetisé"])
            plt.xticks(rotation=90)
            plt.ylabel("Montant Budgetisé")
            plt.title("Intitulé affaire vs Montant Budgetisé")
            st.pyplot(plt)
            plt.clf()

        if len(estimation_nonzero) > 0:
            plt.figure(figsize=(8, 4))
            plt.bar(df_stats["Intitulé affaire"], df_stats["Estimation financière"])
            plt.xticks(rotation=90)
            plt.ylabel("Estimation financière")
            plt.title("Intitulé affaire vs Estimation financière")
            st.pyplot(plt)
            plt.clf()

        # ===============================
        # DIAGRAMME DE DISTRIBUTION
        # ===============================
        st.subheader("📊 Diagrammes de distribution")

        if len(montant_nonzero) > 0:
            plt.figure(figsize=(8, 4))
            sns.histplot(montant_nonzero, kde=True, bins=10, color="skyblue")
            plt.title("Distribution du Montant Budgetisé")
            plt.xlabel("Montant Budgetisé")
            plt.ylabel("Densité")
            st.pyplot(plt)
            plt.clf()

        if len(estimation_nonzero) > 0:
            plt.figure(figsize=(8, 4))
            sns.histplot(estimation_nonzero, kde=True, bins=10, color="salmon")
            plt.title("Distribution de l'Estimation financière")
            plt.xlabel("Estimation financière")
            plt.ylabel("Densité")
            st.pyplot(plt)
            plt.clf()

    else:
        st.warning("⚠️ Aucun résultat trouvé.")
