import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model():
    """Evaluate the trained model"""
    print("🔍 Evaluating Model...")
    
    # Load model and preprocessor
    model_data = joblib.load('../models/best_model_src.pkl')
    preprocessor = joblib.load('../models/preprocessing_src.pkl')
    
    model = model_data['model']
    scaler = preprocessor['scaler']
    
    # Load and prepare data
    df_clean = pd.read_csv('../data/processed/diabetes_cleaned.csv')
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    
    # Split (same as training)
    from sklearn.model_selection import train_test_split
    X_scaled = scaler.transform(X)
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📈 Saved Accuracy: {model_data['accuracy']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../reports/confusion_matrix_src.png')
    plt.show()
    
    print("\n✅ Evaluation completed!")
    return accuracy

if __name__ == "__main__":
    evaluate_model()