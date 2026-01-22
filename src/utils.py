import os
import joblib

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['../data/processed', '../models', '../reports']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Directories created")

def save_model_artifacts(model, preprocessor, accuracy, model_path, preprocessor_path):
    """Save model and preprocessor"""
    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save model with metadata
    model_data = {
        'model': model,
        'accuracy': float(accuracy),
        'features': preprocessor['feature_names'],
        'model_type': type(model).__name__
    }
    joblib.dump(model_data, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Preprocessor saved to: {preprocessor_path}")