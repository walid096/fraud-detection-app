import pandas as pd
import numpy as np
import joblib
import os

# Paths (adjust if needed)
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../models/feature_names.pkl')

# The list of features your model expects (update if needed)
MODEL_FEATURES = [
    'TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts', 'AccountBalance',
    'TransactionAmount_log', 'AccountBalance_log', 'Hour', 'DayOfWeek', 'TimeSinceLastTransaction_s',
    'TransactionType_Credit', 'TransactionType_Debit', 'Channel_ATM', 'Channel_Branch', 'Channel_Online',
    'CustomerOccupation_Doctor', 'CustomerOccupation_Engineer', 'CustomerOccupation_Retired', 'CustomerOccupation_Student'
]

def load_scaler():
    """Load the trained scaler."""
    return joblib.load(SCALER_PATH)

def load_feature_names():
    """Load the list of feature names used during model training."""
    if os.path.exists(FEATURES_PATH):
        return joblib.load(FEATURES_PATH)
    return MODEL_FEATURES

def clean_data(df):
    """Basic cleaning: drop NA, etc."""
    return df.dropna()

def feature_engineering(raw_df):
    """
    Transform raw transaction input into the feature set expected by the model.
    Handles log transforms, one-hot encoding, and missing columns.
    """
    df = raw_df.copy()

    # Rename columns to match training if needed
    rename_map = {
        'montant': 'TransactionAmount',
        'age_client': 'CustomerAge',
        'type_transaction': 'TransactionType',
        'canal': 'Channel',
        'customer_occupation': 'CustomerOccupation',
        # Add more mappings if needed
    }
    df.rename(columns=rename_map, inplace=True)

    # Log transforms (add 1 to avoid log(0))
    df['TransactionAmount_log'] = np.log1p(df['TransactionAmount'])
    if 'AccountBalance' in df.columns:
        df['AccountBalance_log'] = np.log1p(df['AccountBalance'])
    else:
        df['AccountBalance'] = 0
        df['AccountBalance_log'] = 0

    # Example: extract hour and day of week if you have a datetime column
    if 'datetime' in df.columns:
        df['Hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['DayOfWeek'] = pd.to_datetime(df['datetime']).dt.dayofweek
    else:
        df['Hour'] = 0
        df['DayOfWeek'] = 0

    # Fill missing engineered features
    for col in ['TransactionDuration', 'LoginAttempts', 'TimeSinceLastTransaction_s']:
        if col not in df.columns:
            df[col] = 0

    # One-hot encoding for TransactionType, Channel, CustomerOccupation
    cols_to_encode = [col for col in ['TransactionType', 'Channel', 'CustomerOccupation'] if col in df.columns]
    if cols_to_encode:
     df = pd.get_dummies(df, columns=cols_to_encode, prefix=cols_to_encode)
    # Ensure all expected columns are present, add missing with 0
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[MODEL_FEATURES]

    return df

def normalize_transaction(transaction_df, scaler=None):
    """
    Normalize a transaction DataFrame using the trained scaler.
    """
    if scaler is None:
        scaler = load_scaler()
    features = feature_engineering(transaction_df)
    scaled_data = scaler.transform(features)
    return scaled_data

def process_transaction_input(input_dict):
    """
    Accepts a dict (from Streamlit form), returns a DataFrame ready for normalization.
    """
    df = pd.DataFrame([input_dict])
    return feature_engineering(df)