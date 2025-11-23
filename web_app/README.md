# 🌐 Web Dashboard - Sentiment Analysis Transjakarta

Web application interaktif untuk presentasi penelitian sentiment analysis dengan fitur real-time prediction.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd web_app
pip install -r requirements.txt
```

### 2. Train & Save Models

**PENTING**: Jalankan ini terlebih dahulu untuk training models dan menyimpannya!

```bash
python train_and_save_models.py
```

Output:
- Models akan disimpan di folder `models/`
- File yang dibuat:
  - `naive_bayes_model.pkl`
  - `svm_model.pkl`
  - `random_forest_model.pkl`
  - `tfidf_vectorizer.pkl`
  - `model_results.csv`
  - `dataset_stats.json`

### 3. Run Flask Server

```bash
python app.py
```

Buka browser dan akses: **http://localhost:5000**

---

## 📁 Struktur Folder

```
web_app/
│
├── app.py                      # Flask application
├── model_loader.py             # Model loader & preprocessing
├── train_and_save_models.py    # Training script
├── requirements.txt            # Dependencies
│
├── models/                     # Saved models (dibuat setelah training)
│   ├── naive_bayes_model.pkl
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── model_results.csv
│   └── dataset_stats.json
│
├── static/
│   └── css/
│       └── style.css           # Modern CSS
│
└── templates/                  # HTML templates
    ├── index.html              # Landing page
    ├── dashboard.html          # Dashboard dengan charts
    ├── analyzer.html           # Real-time analyzer
    ├── comparison.html         # Model comparison
    └── methodology.html        # Methodology page
```

---

## 🎨 Fitur

### 1. **Landing Page** (`/`)
- Hero section dengan gradient background
- Quick stats (total data, akurasi terbaik)
- Overview fitur
- Model terbaik highlight

### 2. **Dashboard** (`/dashboard`)
- Distribusi sentimen (pie chart)
- Perbandingan akurasi model (bar chart)
- Perbandingan F1 Score (grouped bar chart)
- Interactive charts menggunakan Chart.js

### 3. **Real-time Analyzer** (`/analyzer`)
- Input text area
- Model selector (Naive Bayes / SVM / Random Forest)
- Hasil prediksi dengan:
  - Sentiment label (Positif/Negatif)
  - Confidence score dengan progress bar
  - Probabilities untuk kedua kelas
  - Preprocessing steps visualization
- Contoh cepat untuk testing

### 4. **Model Comparison** (`/comparison`)
- Tabel perbandingan lengkap semua metrik
- Best model highlight
- Detail parameters untuk setiap model
- Metrics: Accuracy, F1 Macro, F1 Weighted, ROC-AUC, CV F1 Mean

### 5. **Methodology** (`/methodology`)
- Pipeline penelitian
- Detail preprocessing steps
- Feature extraction configuration
- Model descriptions
- Hyperparameter yang di-tune
- Referensi

---

## 🔌 API Endpoints

### POST `/api/predict`
Prediksi sentiment untuk input text.

**Request:**
```json
{
  "text": "Transjakarta naik lagi mahal banget",
  "model": "SVM"
}
```

**Response:**
```json
{
  "success": true,
  "sentiment": "Negatif",
  "confidence": 0.95,
  "probabilities": {
    "Negatif": 0.95,
    "Positif": 0.05
  },
  "preprocessed_text": "transjakarta naik mahal",
  "preprocessing_steps": [
    ["Original", "Transjakarta naik lagi mahal banget"],
    ["Cleaned", "transjakarta naik lagi mahal banget"],
    ["Normalized", "transjakarta naik lagi mahal banget"],
    ["Stopwords Removed", "transjakarta naik mahal"],
    ["Stemmed", "transjakarta naik mahal"]
  ]
}
```

### GET `/api/stats`
Dataset statistics.

**Response:**
```json
{
  "total_samples": 2757,
  "negative_samples": 1500,
  "positive_samples": 1257,
  "test_samples": 551,
  "train_samples": 3304
}
```

### GET `/api/models`
Model information.

**Response:**
```json
[
  {
    "Model": "SVM",
    "Best_Params": "{'C': 10, 'kernel': 'linear', 'gamma': 'scale'}",
    "CV_F1_Mean": 0.8421,
    "Test_Accuracy": 0.8634,
    "Test_F1_Macro": 0.8542,
    "Test_F1_Weighted": 0.8598,
    "Test_ROC_AUC": 0.9123
  },
  ...
]
```

---

## 🎨 Design Features

- **Modern UI/UX**: Gradient backgrounds, glassmorphism effects
- **Transjakarta Theme**: Warna biru & hijau sesuai branding Transjakarta
- **Responsive**: Mobile-friendly design
- **Smooth Animations**: Fade in, slide up, loading states
- **Interactive Charts**: Chart.js untuk visualisasi data
- **Real-time Updates**: AJAX untuk prediksi tanpa reload page

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Charts**: Chart.js
- **ML Libraries**: scikit-learn, Sastrawi, demoji
- **Data**: pandas, numpy

---

## 📝 Notes

1. **Training Time**: Training dengan GridSearchCV memakan waktu ~5-15 menit tergantung spesifikasi komputer
2. **Model Size**: Total ukuran models ~10-20 MB
3. **Browser Support**: Chrome, Firefox, Safari, Edge (modern browsers)
4. **Port**: Default port 5000, bisa diubah di `app.py`

---

## 🐛 Troubleshooting

### Error: "Models not found"
**Solusi**: Jalankan `python train_and_save_models.py` terlebih dahulu

### Error: "Module not found"
**Solusi**: Install dependencies dengan `pip install -r requirements.txt`

### Charts tidak muncul
**Solusi**: Pastikan koneksi internet aktif (Chart.js loaded dari CDN)

### Prediksi lambat
**Solusi**: Normal untuk first prediction (model loading). Prediksi berikutnya akan lebih cepat.

---

## 📞 Support

Untuk pertanyaan atau issue, silakan hubungi author.

---

**Last Updated**: November 2025
