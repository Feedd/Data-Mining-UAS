# 🏥 Sistem Prediksi Diabetes - Capstone Project

Sistem prediksi diabetes menggunakan machine learning untuk tugas kuliah Data Mining.

## 📁 Struktur Project

capstone-project-data-mining/
│
├── data/
│ ├── raw/
│ ├── processed/
│ │ └── mental_health_clean.csv
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_modeling.ipynb
│ └── 03_interpretation.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── utils.py
│
├── models/
│ ├── best_model.pkl
│ └── preprocessing.pkl
│
├── app/
│ └── app.py
│
├── reports/
│ ├── final_report.pdf
│ └── presentation.pptx
│
├── requirements.txt
└── README.md

## 🚀 Cara Pakai

### 1. Install Dependencies
```bash
pip install -r requirements.txt

2. Jalankan Notebooks (Urut)
01_eda.ipynb - Analisis data

02_modeling.ipynb - Training model

03_interpretation.ipynb - Interpretasi hasil

📊 Hasil Model
Akurasi: ~85%
Algoritma: Random Forest
Dataset: Pima Indians Diabetes (768 data)

Fitur Penting:
1.Glucose Level (paling penting)
2.BMI
3.Age
4.Diabetes Pedigree Function

🎯 Fitur Aplikasi
1.Prediksi Real-time - Input data pasien, langsung dapat prediksi
2.Analisis Data - Visualisasi dataset
3.Info Model - Lihat performa model

👤 Pengembang
Nama: Pasha Aditya Dhananjaya
NIM : A11.2023.15399
Mata Kuliah: Data Mining
