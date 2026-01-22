# 🏥 Sistem Prediksi Diabetes - Capstone Project

Sistem prediksi diabetes menggunakan machine learning untuk tugas kuliah Data Mining.

## 📁 Struktur Project

capstone-project-data-mining/
│
├── data/
│ ├── raw/ # Data mentah
│ ├── processed/ # Data yang sudah diproses
│ └── external/ # Data referensi eksternal
├── notebooks/
│ ├── 01_eda.ipynb # EDA dan preprocessing
│ ├── 02_modeling.ipynb # Pemodelan dan evaluasi
│ └── 03_interpretation.ipynb # Interpretasi model
├── src/
│ ├── data_preprocessing.py # Script preprocessing
│ ├── train_model.py # Script training
│ ├── evaluate_model.py # Script evaluasi
│ └── utils.py # Fungsi utilitas
├── models/
│ ├── best_model.pkl # Model terbaik
│ └── preprocessing.pkl # Pipeline preprocessing
├── app/
│ ├── app.py # Aplikasi Streamlit utama
│ ├── pages/ # Halaman tambahan Streamlit
│ └── assets/ # Gambar, CSS, dll.
├── reports/
│ ├── final_report.pdf # Laporan akhir
│ └── presentation.pptx # Slide presentasi
├── requirements.txt # Dependencies
├── README.md # Dokumentasi proyek

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
