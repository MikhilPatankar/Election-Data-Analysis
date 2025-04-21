import joblib
import pandas as pd
import numpy as np
# Import preprocessors - needed for loading objects and potentially type checks
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# --- Configuration ---
# !!! UPDATE THESE FILENAMES if they are different from how they were saved !!!
MODEL_FILENAME = 'final_model.joblib' # Example filename, adjust if needed
SCALER_FILENAME = 'min_max_scaler.joblib'
ENCODER_FILENAME = 'ordinal_encoder.joblib'

# Define the feature columns in the exact order the model expects
# (Should match the 'feature_columns' list from the training script)
FEATURE_COLUMNS = ['STATE', 'PARTY', 'GENDER', 'CRIMINALCASES', 'AGE', 'CATEGORY', 'EDUCATION', 'ASSETS', 'LIABILITIES', 'TOTAL ELECTORS']
CATEGORICAL_FEATURES = ['STATE', 'PARTY', 'GENDER', 'CATEGORY', 'EDUCATION']
NUMERICAL_FEATURES = ['CRIMINALCASES', 'AGE', 'ASSETS', 'LIABILITIES', 'TOTAL ELECTORS']

# --- Load Model and Preprocessors ---
try:
    print(f"Loading model from {MODEL_FILENAME}...")
    model = joblib.load(MODEL_FILENAME)
    print("Model loaded successfully.")

    print(f"Loading scaler from {SCALER_FILENAME}...")
    scaler = joblib.load(SCALER_FILENAME)
    print("Scaler loaded successfully.")

    print(f"Loading encoder from {ENCODER_FILENAME}...")
    encoder = joblib.load(ENCODER_FILENAME)
    print("Encoder loaded successfully.")

except FileNotFoundError as e:
    print(f"\nError loading file: {e}")
    print("Please ensure the model, scaler, and encoder files (.joblib) exist in the same directory as this script.")
    exit()
except Exception as e:
    print(f"\nAn error occurred during loading: {e}")
    exit()

# --- Prepare New Data for Prediction ---
# Example: Create hypothetical new data for prediction
# Replace this with your actual new data
new_data_dict = {
    'STATE': ['Maharashtra', 'Gujarat', 'West Bengal'],
    'PARTY': ['BJP', 'INC', 'AITC'], # Use parties the model was trained on or 'Other' if needed
    'GENDER': ['MALE', 'FEMALE', 'FEMALE'],
    'CRIMINALCASES': [1, 0, 2],
    'AGE': [55.0, 45.0, 60.0],
    'CATEGORY': ['GENERAL', 'SC', 'GENERAL'],
    'EDUCATION': ['Graduate', 'Post Graduate', '12th Pass'], # Use education levels model saw
    'ASSETS': [5000000.0, 2000000.0, 10000000.0],
    'LIABILITIES': [100000.0, 50000.0, 500000.0],
    'TOTAL ELECTORS': [1800000, 1600000, 1750000]
}
new_candidates_df_raw = pd.DataFrame(new_data_dict)

print("\n--- Raw New Data ---")
print(new_candidates_df_raw)

# --- Preprocess New Data ---
# Create a copy to avoid modifying the raw data DataFrame
new_candidates_df_processed = new_candidates_df_raw.copy()

try:
    print("\nPreprocessing new data...")
    # Ensure correct dtypes for categorical features before encoding
    for cat in CATEGORICAL_FEATURES:
        new_candidates_df_processed[cat] = new_candidates_df_processed[cat].astype('category')

    # Apply the *LOADED* encoder
    new_candidates_df_processed[CATEGORICAL_FEATURES] = encoder.transform(new_candidates_df_processed[CATEGORICAL_FEATURES])
    print("Categorical encoding applied.")

    # Apply the *LOADED* scaler
    new_candidates_df_processed[NUMERICAL_FEATURES] = scaler.transform(new_candidates_df_processed[NUMERICAL_FEATURES])
    print("Numerical scaling applied.")

    # Ensure columns are in the correct order (if needed, though processing by name handles this)
    new_candidates_df_processed = new_candidates_df_processed[FEATURE_COLUMNS]

    print("\n--- Preprocessed New Data (Ready for Prediction) ---")
    print(new_candidates_df_processed)

    # --- Make Predictions ---
    print("\nMaking predictions...")
    new_predictions = model.predict(new_candidates_df_processed)
    print("Predictions completed.")

    # --- Display Results ---
    # Add predictions back to the raw data DataFrame for easy interpretation
    new_candidates_df_raw['PREDICTED_WINNER (0=Loss, 1=Win)'] = new_predictions

    print("\n--- Predictions for New Candidates ---")
    print(new_candidates_df_raw[['NAME' if 'NAME' in new_candidates_df_raw else 'STATE', 'PARTY', 'PREDICTED_WINNER (0=Loss, 1=Win)']]) # Show key info + prediction

    # Optional: Get prediction probabilities if the model supports it
    if hasattr(model, "predict_proba"):
        print("\nCalculating prediction probabilities...")
        prediction_probabilities = model.predict_proba(new_candidates_df_processed)
        print("Probabilities calculated.")
        # Add probabilities to the DataFrame
        new_candidates_df_raw['Probability_Loss (Class 0)'] = prediction_probabilities[:, 0]
        new_candidates_df_raw['Probability_Win (Class 1)'] = prediction_probabilities[:, 1]
        print("\n--- Predictions with Probabilities ---")
        print(new_candidates_df_raw[['STATE', 'PARTY', 'PREDICTED_WINNER (0=Loss, 1=Win)', 'Probability_Win (Class 1)']])

except Exception as e:
    print(f"\nAn error occurred during preprocessing or prediction: {e}")
    print("Ensure the new data has the correct columns and data types, and that categories/values are consistent with the training data.")

