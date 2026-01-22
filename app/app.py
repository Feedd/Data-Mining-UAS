import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# ====================== AUTO-TRAIN MODEL ======================
def auto_train_model():
    """Auto train model if not exists - untuk deploy ke cloud"""
    model_path = Path('models/best_model.pkl')
    data_path = Path('data/processed/diabetes_cleaned.csv')
    
    # Jika model sudah ada, skip
    if model_path.exists():
        return True
    
    # Jika data cleaned belum ada, preprocess dulu
    if not data_path.exists():
        st.info("🔧 Preprocessing data...")
        preprocess_data()
    
    st.info("🤖 Training model (first time setup)...")
    
    try:
        # Import modul yang diperlukan
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        
        # Load data
        df = pd.read_csv('data/processed/diabetes_cleaned.csv')
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
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
        
        # Save model
        model_data = {
            'model': model,
            'accuracy': float(accuracy),
            'features': X.columns.tolist(),
            'model_type': 'RandomForestClassifier'
        }
        
        # Buat folder models jika belum ada
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_data, 'models/best_model.pkl')
        joblib.dump(preprocessor, 'models/preprocessing.pkl')
        
        st.success(f"✅ Model trained! Accuracy: {accuracy:.1%}")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to train model: {str(e)}")
        return False

def preprocess_data():
    """Auto preprocess data"""
    try:
        # Buat folder
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Dataset URL (Pima Indians Diabetes)
        # Untuk deploy, kita perlu punya file diabetes.csv di repo
        # Atau download otomatis:
        import urllib.request
        
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        raw_data = pd.read_csv(url, header=None)
        
        # Beri nama kolom
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        raw_data.columns = columns
        
        # Simpan raw data
        raw_data.to_csv('data/raw/diabetes.csv', index=False)
        
        # Cleaning data
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in cols_with_zeros:
            raw_data[col] = raw_data[col].replace(0, np.nan)
            raw_data[col] = raw_data[col].fillna(raw_data[col].median())
        
        # Scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = raw_data.drop('Outcome', axis=1)
        y = raw_data['Outcome']
        X_scaled = scaler.fit_transform(X)
        
        # Save cleaned data
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        df_clean = X_scaled_df.copy()
        df_clean['Outcome'] = y.values
        df_clean.to_csv('data/processed/diabetes_cleaned.csv', index=False)
        
        return True
        
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return False

# ====================== APP CODE ======================
# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Auto setup saat pertama kali run
if 'model_loaded' not in st.session_state:
    with st.spinner("Setting up application..."):
        auto_train_model()
        st.session_state.model_loaded = True

# Load model function
@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        # Coba load model
        model_paths = ['models/best_model.pkl', 'models/best_model_src.pkl']
        
        for path in model_paths:
            if Path(path).exists():
                model_data = joblib.load(path)
                preprocessor = joblib.load(path.replace('best_model', 'preprocessing'))
                return model_data, preprocessor
        
        # Jika tidak ada, coba train
        auto_train_model()
        model_data = joblib.load('models/best_model.pkl')
        preprocessor = joblib.load('models/preprocessing.pkl')
        return model_data, preprocessor
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

# UI SAMA SEPERTI SEBELUMNYA (tapi path relative)
with st.sidebar:
    st.title("🏥 Diabetes Predictor")
    st.markdown("---")
    
    model_data, _ = load_model()
    if model_data:
        st.metric("Model Accuracy", f"{model_data.get('accuracy', 0):.1%}")
    
    st.markdown("---")
    st.info("Enter patient details and click Predict")

# Main app
st.title("🏥 Diabetes Prediction System")
st.markdown("Predict diabetes risk based on patient health metrics")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analysis", "ℹ️ About"])

with tab1:
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 3)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    
    with col2:
        insulin = st.number_input("Insulin (μU/mL)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, 0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree", 0.0, 2.5, 0.5, 0.01)
        age = st.slider("Age", 0, 100, 30)
    
    if st.button("Predict Diabetes Risk", type="primary"):
        model_data, preprocessor = load_model()
        
        if model_data and preprocessor:
            with st.spinner("Analyzing..."):
                features = np.array([[pregnancies, glucose, blood_pressure, 
                                    skin_thickness, insulin, bmi,
                                    diabetes_pedigree, age]])
                
                scaler = preprocessor['scaler']
                features_scaled = scaler.transform(features)
                
                model = model_data['model']
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
                
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if prediction == 1:
                        st.error("## 🚨 HIGH RISK: Diabetes")
                    else:
                        st.success("## ✅ LOW RISK: No Diabetes")
                    
                    confidence = max(probability) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col_res2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(['No Diabetes', 'Diabetes'], probability, 
                          color=['green', 'red'])
                    ax.set_ylabel('Probability')
                    ax.set_ylim(0, 1)
                    
                    for i, v in enumerate(probability):
                        ax.text(i, v + 0.02, f'{v:.1%}', 
                               ha='center', fontweight='bold')
                    
                    st.pyplot(fig)
        else:
            st.error("Model not loaded")

with tab2:
    st.header("Data Analysis")
    
    try:
        if Path('data/processed/diabetes_cleaned.csv').exists():
            df = pd.read_csv('data/processed/diabetes_cleaned.csv')
            
            st.subheader("Dataset Info")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total", len(df))
            with col2: st.metric("Diabetic", df['Outcome'].sum())
            with col3: st.metric("Non-Diabetic", len(df) - df['Outcome'].sum())
            with col4: st.metric("Diabetes Rate", f"{df['Outcome'].mean()*100:.1f}%")
            
            # Simple visualization
            st.subheader("Glucose Distribution")
            fig, ax = plt.subplots()
            ax.hist(df['Glucose'], bins=20, edgecolor='black')
            ax.set_xlabel('Glucose Level')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        else:
            st.info("No data available. Please run prediction first.")
    except:
        st.info("Data not available yet")

with tab3:
    st.header("About This App")
    
    st.markdown("""
    ### 🏥 Diabetes Prediction App
    
    **Features:**
    - Real-time diabetes risk prediction
    - Based on 8 health metrics
    - Machine learning model (Random Forest)
    
    **Data:** Pima Indians Diabetes Dataset
    
    **Note:** For educational purposes only.
    """)

st.markdown("---")
st.caption("Deployed on Streamlit Cloud | Data Mining Capstone Project")