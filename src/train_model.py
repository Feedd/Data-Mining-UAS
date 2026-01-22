import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from utils import setup_directories, save_model_artifacts

def train_model():
    """Train model - sama dengan notebook 2"""
    setup_directories()
    
    print("🤖 Training Model...")
    
    # Load data
    df_clean = pd.read_csv('../data/processed/diabetes_cleaned.csv')
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    
    # Create preprocessor
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    preprocessor = {
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'n_features': X.shape[1]
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save artifacts
    save_model_artifacts(
        model=model,
        preprocessor=preprocessor,
        accuracy=accuracy,
        model_path='../models/best_model_src.pkl',
        preprocessor_path='../models/preprocessing_src.pkl'
    )
    
    print("\n✅ Training completed!")
    return model, accuracy

if __name__ == "__main__":
    train_model()