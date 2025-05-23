import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_transaction_history(history_df):
    """
    Plot the transaction history with fraud/normal labels.
    """
    if history_df.empty:
        st.info("Aucune transaction à afficher.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(
        data=history_df,
        x=history_df.index,
        y="risk_score",
        hue="suspect",
        palette={True: "red", False: "green"},
        ax=ax
    )
    ax.set_title("Historique des transactions et scores de risque")
    ax.set_xlabel("Transaction #")
    ax.set_ylabel("Score de risque")
    st.pyplot(fig)

def plot_detection_stats(history_df):
    """
    Show statistics: number of normal vs. suspect transactions.
    """
    if history_df.empty:
        st.info("Aucune statistique à afficher.")
        return

    stats = history_df['suspect'].value_counts().rename({True: "Suspecte", False: "Normale"})
    fig, ax = plt.subplots()
    stats.plot(kind='bar', color=['red', 'green'], ax=ax)
    ax.set_title("Statistiques de détection")
    ax.set_ylabel("Nombre de transactions")
    st.pyplot(fig)

def plot_risk_distribution(history_df):
    """
    Plot the distribution of risk scores.
    """
    if history_df.empty:
        st.info("Aucune distribution à afficher.")
        return

    fig, ax = plt.subplots()
    sns.histplot(history_df['risk_score'], bins=20, kde=True, ax=ax, color="blue")
    ax.set_title("Distribution des scores de risque")
    ax.set_xlabel("Score de risque")
    st.pyplot(fig)

def show_transaction_table(history_df):
    """
    Display the transaction history as a table.
    """
    if history_df.empty:
        st.info("Aucune transaction à afficher.")
        return
    st.dataframe(history_df)

def export_results(history_df, filename="historique_transactions.csv"):
    """
    Export the transaction history to a CSV file.
    """
    if history_df.empty:
        st.warning("Aucune donnée à exporter.")
        return
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Exporter l'historique en CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )