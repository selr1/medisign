import pandas as pd
import numpy as np
import os

# Path to feature matrix
DATA_PATH = '../CSVs/preprocessed.csv'

def validate():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    print("--- Dataset Structure ---")
    # Check total samples and features
    print(f"Total samples: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]} (Expected 64: 63 features + 1 label)")

    print("\n--- Integrity Checks ---")
    # Look for empty cells
    null_count = df.isnull().sum().sum()
    print(f"Missing values: {null_count}")

    # Verify every row has 63 numerical features
    feature_cols = df.iloc[:, :-1]
    is_numeric = np.all(feature_cols.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
    print(f"All features numerical: {is_numeric}")

    print("\n--- Class Distribution ---")
    # List how many images were processed per folder/label
    counts = df['label'].value_counts()
    print(counts)

    # Warning for low data
    low_data = counts[counts < 10].index.tolist()
    if low_data:
        print(f"\nWarning: Low sample count (<10) for classes: {low_data}")

    print("\nValidation complete.")

if __name__ == "__main__":
    validate()
