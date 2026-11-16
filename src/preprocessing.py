import pandas as pd
import os

def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    return pd.read_csv(csv_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with any missing values (none expected in provided dataset)
    df = df.dropna()
    # Encode categorical columns
    if 'Product ID' in df.columns:
        df['Product ID'] = df['Product ID'].astype('category').cat.codes
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype('category').cat.codes
    return df

def save_processed(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

def main():
    raw_path = os.path.join('data', 'raw', 'ai4i2020.csv')
    processed_path = os.path.join('data', 'processed', 'processed.csv')
    df_raw = load_raw(raw_path)
    df_clean = clean_data(df_raw)
    save_processed(df_clean, processed_path)
    print(f"Processed data saved to {processed_path}")
    
if __name__ == "__main__":
    main()