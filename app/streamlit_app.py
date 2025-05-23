import sys
import os
import streamlit as st
import pandas as pd

# Ensure utils are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_processing import (
    load_scaler, process_transaction_input, normalize_transaction, MODEL_FEATURES
)
from utils.model_utils import (
    load_model, evaluate_transaction
)
from utils.visualization import (
    plot_transaction_history, plot_detection_stats, plot_risk_distribution,
    show_transaction_table, export_results
)

# Load model and scaler once
model = load_model()
scaler = load_scaler()
feature_names = MODEL_FEATURES  # Or load_feature_names() if you saved them

# Session state for transaction history
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(
        columns=['Transaction #', 'risk_score', 'suspect', 'reasons'] + feature_names
    )
if 'counter' not in st.session_state:
    st.session_state['counter'] = 1

st.title("Détection de Fraude - Vérification Transaction")

# --- User Input Form ---
with st.form("transaction_form"):
    montant = st.number_input("Montant", min_value=0.0)
    type_transaction = st.selectbox("Type de transaction", ["Débit", "Crédit"])
    localisation = st.text_input("Localisation")
    canal = st.selectbox("Canal", ["ATM", "Online", "Branch"])
    age_client = st.number_input("Âge du client", min_value=0)
    # Add more fields as needed (e.g., occupation, account balance, etc.)
    occupation = st.selectbox("Profession du client", ["Doctor", "Engineer", "Retired", "Student"])
    submitted = st.form_submit_button("Vérifier la Transaction")

if submitted:
    # Prepare input dict
    user_input = {
        'montant': montant,
        'type_transaction': type_transaction,
        'localisation': localisation,
        'canal': canal,
        'age_client': age_client,
        'customer_occupation': occupation,
        # Add more fields as needed
    }
    # Feature engineering
    transaction_df = process_transaction_input(user_input)
    # Scaling
    transaction_scaled = normalize_transaction(transaction_df, scaler)
    # Model evaluation
    suspect, risk_score, reasons = evaluate_transaction(
        model, transaction_scaled, feature_names
    )[:3]

    # Show result
    if suspect:
        st.error(f"Transaction Suspecte ! Score de risque : {risk_score:.2f}")
        st.write("Raisons principales :")
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.success(f"Transaction Normale. Score de risque : {risk_score:.2f}")

    # Save to history
    history_row = {
        'Transaction #': st.session_state['counter'],
        'risk_score': risk_score,
        'suspect': suspect,
        'reasons': "; ".join(reasons) if reasons else "",
    }
    for i, col in enumerate(feature_names):
        history_row[col] = transaction_df.iloc[0][col]
    st.session_state['history'] = pd.concat(
        [st.session_state['history'], pd.DataFrame([history_row])],
        ignore_index=True
    )
    st.session_state['counter'] += 1

# --- Tabs for History, Stats, Export ---
st.header("Analyse et Historique")
tab1, tab2, tab3, tab4 = st.tabs([
    "Historique", "Statistiques", "Distribution des risques", "Exporter"
])

with tab1:
    show_transaction_table(st.session_state['history'])

with tab2:
    plot_detection_stats(st.session_state['history'])

with tab3:
    plot_risk_distribution(st.session_state['history'])
    plot_transaction_history(st.session_state['history'])

with tab4:
    export_results(st.session_state['history'])

# Optionally: allow batch verification (CSV upload)
st.sidebar.header("Vérification par lot")
uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV de transactions", type="csv")
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    results = []
    for idx, row in batch_df.iterrows():
        transaction_df = process_transaction_input(row.to_dict())
        transaction_scaled = normalize_transaction(transaction_df, scaler)
        suspect, risk_score, reasons = evaluate_transaction(
            model, transaction_scaled, feature_names
        )[:3]
        results.append({
            **row.to_dict(),
            'risk_score': risk_score,
            'suspect': suspect,
            'reasons': "; ".join(reasons) if reasons else ""
        })
    results_df = pd.DataFrame(results)
    st.sidebar.success("Vérification terminée !")
    st.sidebar.dataframe(results_df)
    export_results(results_df, filename="resultats_batch.csv")