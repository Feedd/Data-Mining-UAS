import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import setup_directories

def preprocess_data():
    """Preprocess diabetes data - sama dengan notebook 1"""
    setup_directories()
    
    print("📊 Loading data...")
    df = pd.read_csv('../data/raw/diabetes.csv')
    print(f"   Original shape: {df.shape}")
    
    # Columns with zeros
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace 0 with NaN then fill with median
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    # Scale features
    scaler = StandardScaler()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_scaled = scaler.fit_transform(X)
    
    # Create cleaned dataframe
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    df_clean = X_scaled_df.copy()
    df_clean['Outcome'] = y.values
    
    # Save to CSV
    df_clean.to_csv('../data/processed/diabetes_cleaned.csv', index=False)
    
    print(f"✅ Cleaned data saved: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    preprocess_data()