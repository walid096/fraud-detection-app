import joblib
import numpy as np
import os

# Path to your trained model (adjust if needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/kmeans_model.pkl')

def load_model():
    """Load the trained clustering model."""
    return joblib.load(MODEL_PATH)

def predict_cluster(model, transaction_scaled):
    """Predict the cluster for the transaction."""
    return model.predict(transaction_scaled)

def calculate_distance_to_centroid(model, transaction_scaled):
    """
    Calculate the Euclidean distance from the transaction to its cluster centroid.
    """
    cluster = predict_cluster(model, transaction_scaled)[0]
    centroid = model.cluster_centers_[cluster]
    distance = np.linalg.norm(transaction_scaled - centroid)
    return distance, cluster

def is_transaction_suspect(distance, threshold=2.0):
    """
    Decide if the transaction is suspect based on the distance to the centroid.
    The threshold should be set based on validation.
    """
    return distance > threshold

def get_suspicion_reasons(transaction_scaled, model, cluster):
    """
    Provide reasons for suspicion: which features are most different from the centroid.
    Returns a list of (feature_index, difference) sorted by absolute difference.
    """
    centroid = model.cluster_centers_[cluster]
    diffs = np.abs(transaction_scaled[0] - centroid)
    # Get indices of top 3 most different features
    top_indices = np.argsort(diffs)[::-1][:3]
    return top_indices, diffs[top_indices]

def explain_suspicion(transaction_scaled, model, cluster, feature_names):
    """
    Return a human-readable explanation of why the transaction is suspicious.
    """
    top_indices, top_diffs = get_suspicion_reasons(transaction_scaled, model, cluster)
    reasons = []
    for idx, diff in zip(top_indices, top_diffs):
        reasons.append(f"{feature_names[idx]} (écart: {diff:.2f})")
    return reasons

def evaluate_transaction(model, transaction_scaled, feature_names, threshold=2.0):
    """
    Full pipeline: predict, score, and explain.
    Returns: suspect (bool), risk_score (float), reasons (list of str)
    """
    distance, cluster = calculate_distance_to_centroid(model, transaction_scaled)
    suspect = is_transaction_suspect(distance, threshold)
    reasons = []
    if suspect:
        reasons = explain_suspicion(transaction_scaled, model, cluster, feature_names)
    return suspect, distance, reasons