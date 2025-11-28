Web Dashboard untuk Sentiment Analysis Transjakarta
Membuat aplikasi web interaktif untuk presentasi penelitian sentiment analysis dengan fitur real-time prediction dan visualisasi komprehensif.

User Review Required
IMPORTANT

Design Decisions:

Menggunakan Flask (Python) untuk backend agar mudah integrasi dengan model ML yang sudah ada
Frontend menggunakan vanilla HTML/CSS/JavaScript (tanpa framework) untuk simplicity
Model akan di-load saat startup untuk performa optimal
Real-time prediction akan menggunakan AJAX untuk user experience yang smooth
NOTE

Fitur Utama:

Dashboard dengan visualisasi hasil penelitian
Real-time sentiment analyzer (input text → prediksi)
Model comparison page
About/Methodology page
Proposed Changes
Backend (Flask Application)
[NEW] 
app.py
Flask application dengan endpoints:

GET / - Landing page
GET /dashboard - Dashboard dengan visualisasi
GET /analyzer - Real-time sentiment analyzer
POST /api/predict - API endpoint untuk prediksi sentiment
GET /api/stats - API endpoint untuk statistik dataset
GET /comparison - Model comparison page
GET /methodology - About/Methodology page
Akan load:

Trained models (best model dari hasil training)
TF-IDF vectorizer
Preprocessing functions dari 
preposseing.py
[NEW] 
model_loader.py
Utility untuk:

Load trained models
Load TF-IDF vectorizer
Preprocessing pipeline
Prediction function
[NEW] 
train_and_save_models.py
Script untuk training dan menyimpan models:

Modifikasi dari 
pemodelan.py
 yang sudah ada
Save best models ke file .pkl
Save TF-IDF vectorizer
Save model comparison results
Frontend (HTML/CSS/JavaScript)
[NEW] 
index.html
Landing page dengan:

Hero section dengan gradient background
Overview penelitian
Quick stats (jumlah data, akurasi model terbaik)
Navigation ke halaman lain
Modern design dengan animations
[NEW] 
dashboard.html
Dashboard dengan visualisasi:

Distribusi sentimen (pie chart)
Model comparison (bar chart)
Confusion matrix (heatmap)
Top bigrams (word frequency)
Interactive charts menggunakan Chart.js
[NEW] 
analyzer.html
Real-time sentiment analyzer:

Text input area
Model selector (Naive Bayes / SVM / Random Forest)
Predict button
Result display dengan:
Sentiment label (Positif/Negatif)
Confidence score
Preprocessing steps visualization
Animation untuk hasil
[NEW] 
comparison.html
Model comparison page:

Table dengan semua metrik (Accuracy, F1, ROC-AUC)
Best parameters untuk setiap model
CV scores dengan error bars
Confusion matrices side-by-side
[NEW] 
methodology.html
About/Methodology page:

Flow diagram preprocessing
Hyperparameter yang digunakan
Dataset information
References
[NEW] 
style.css
Modern CSS dengan:

Gradient backgrounds
Glassmorphism effects
Smooth animations
Responsive design
Dark mode support (optional)
Custom color palette (biru/hijau untuk Transjakarta theme)
[NEW] 
main.js
JavaScript untuk:

AJAX calls ke API
Chart rendering (Chart.js)
Form handling
Animations
Loading states
Configuration
[NEW] 
requirements.txt
Dependencies untuk web app:

Flask
scikit-learn
pandas
numpy
joblib (untuk save/load models)
Verification Plan
Automated Tests
Tidak ada automated tests untuk web app ini (fokus pada demo/presentasi).

Manual Verification
Training & Save Models

cd "d:/STIS/Semester 5/Information Retrieval/Perkuliahan/Praktikum/Tugas Akhir Final/web_app"
python train_and_save_models.py
Verifikasi: File .pkl untuk models dan vectorizer terbuat di folder models/
Start Flask Server

cd "d:/STIS/Semester 5/Information Retrieval/Perkuliahan/Praktikum/Tugas Akhir Final/web_app"
python app.py
Verifikasi: Server running di http://localhost:5000
Tidak ada error saat load models
Test Landing Page

Buka browser ke http://localhost:5000
Verifikasi: Landing page tampil dengan design modern
Verifikasi: Navigation links berfungsi
Test Dashboard

Klik menu "Dashboard" atau akses http://localhost:5000/dashboard
Verifikasi: Charts tampil dengan benar (pie chart, bar chart)
Verifikasi: Data sesuai dengan hasil penelitian
Test Real-time Analyzer

Klik menu "Analyzer" atau akses http://localhost:5000/analyzer
Input contoh text: "Transjakarta naik lagi mahal banget"
Pilih model (misal: SVM)
Klik "Predict"
Verifikasi: Hasil prediksi muncul (Negatif/Positif)
Verifikasi: Confidence score ditampilkan
Verifikasi: Preprocessing steps terlihat
Test Model Comparison

Akses http://localhost:5000/comparison
Verifikasi: Table comparison tampil dengan semua metrik
Verifikasi: Confusion matrices tampil untuk semua model
Test Methodology Page

Akses http://localhost:5000/methodology
Verifikasi: Flow diagram dan informasi metodologi tampil
Responsive Design

Resize browser window ke ukuran mobile
Verifikasi: Layout tetap rapi dan readable
Verifikasi: Navigation menu responsive (hamburger menu)
Performance

Test prediksi dengan berbagai panjang text
Verifikasi: Response time < 2 detik
Verifikasi: Tidak ada lag saat render charts
