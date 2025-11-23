# 📊 Sentiment Analysis: Kenaikan Tarif Transjakarta

Analisis sentimen masyarakat terhadap rencana kenaikan tarif Transjakarta menggunakan data dari TikTok dan YouTube dengan metode **Naive Bayes**, **SVM**, dan **Random Forest**.

---

## 📁 Struktur Proyek

```
Tugas Akhir Final/
│
├── data_labeling_tj_tiktokyt_final.csv    # Data mentah dengan label manual
├── data_siap_model_2class.csv             # Data hasil preprocessing
│
├── preposseing.py                         # Script preprocessing teks
├── EVD dan N-gram.py                      # Exploratory Data Analysis & Visualisasi
├── pemodelan.py                           # Training & evaluasi model
│
├── model_comparison_results.csv           # Hasil perbandingan model (output)
├── model_comparison_chart.png             # Grafik perbandingan (output)
│
├── ANALISIS_PENELITIAN.md                 # Analisis lengkap penelitian
└── README.md                              # File ini
```

---

## 🚀 Cara Menjalankan

### 1. **Preprocessing Data**

Jalankan script preprocessing untuk membersihkan data mentah:

```bash
python preposseing.py
```

**Output**: `data_siap_model_2class.csv`

**Proses yang dilakukan**:
- Cleaning (hapus emoji, URL, mention, angka)
- Normalisasi slang/alay
- Stopword removal (dengan mempertahankan kata negasi)
- Stemming menggunakan Sastrawi

---

### 2. **Exploratory Data Analysis (EDA)**

Jalankan script untuk visualisasi data:

```bash
python "EVD dan N-gram.py"
```

**Visualisasi yang dihasilkan**:
- Distribusi sentimen (bar chart)
- Word cloud untuk sentimen Negatif dan Positif
- Distribusi panjang teks per sentimen
- Top 15 bigram untuk masing-masing sentimen

---

### 3. **Pemodelan & Evaluasi**

Jalankan script pemodelan dengan hyperparameter tuning:

```bash
python pemodelan.py
```

**Proses yang dilakukan**:
- TF-IDF vectorization dengan bigram (1,2)
- Train-test split (80:20) dengan stratified sampling
- SMOTE untuk balancing data
- **GridSearchCV** untuk hyperparameter tuning (5-fold CV)
- Evaluasi dengan multiple metrics:
  - Accuracy
  - F1 Macro Score
  - F1 Weighted Score
  - ROC-AUC Score
  - Confusion Matrix

**Output**:
- `model_comparison_results.csv` - Tabel hasil evaluasi
- `model_comparison_chart.png` - Grafik perbandingan model
- Confusion matrix untuk setiap model

---

## 📦 Dependencies

Install semua library yang diperlukan:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn demoji sastrawi wordcloud swifter
```

**Library utama**:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `imbalanced-learn` - SMOTE
- `Sastrawi` - Stemming Bahasa Indonesia
- `demoji` - Emoji removal
- `wordcloud` - Word cloud visualization

---

## 🎯 Metodologi

### 1. **Data Collection**
- Sumber: Komentar TikTok dan YouTube
- Total data: ~2,757 komentar (setelah preprocessing)
- Klasifikasi: Binary (Positif vs Negatif)

### 2. **Preprocessing**
- Text cleaning (emoji, URL, mention removal)
- Slang normalization (kamus alay khusus Transjakarta)
- Stopword removal (mempertahankan kata negasi)
- Stemming (Sastrawi)

### 3. **Feature Extraction**
- TF-IDF Vectorizer
- N-gram: (1, 2) - unigram dan bigram
- max_features: 3000
- min_df: 2 (kata muncul minimal 2x)
- max_df: 0.8 (kata muncul maksimal di 80% dokumen)

### 4. **Modeling**
Tiga algoritma machine learning:

| Model | Hyperparameter yang di-tune |
|-------|------------------------------|
| **Naive Bayes** | alpha: [0.1, 0.5, 1.0, 2.0, 5.0] |
| **SVM** | C: [0.1, 1, 10, 100]<br>kernel: ['linear', 'rbf']<br>gamma: ['scale', 'auto'] |
| **Random Forest** | n_estimators: [50, 100, 200]<br>max_depth: [10, 20, None]<br>min_samples_split: [2, 5, 10] |

### 5. **Evaluation**
- **Cross-Validation**: 5-fold CV pada training set
- **Metrics**: Accuracy, F1 Macro, F1 Weighted, ROC-AUC
- **Confusion Matrix**: Visualisasi untuk setiap model

---

## 📊 Hasil (Contoh)

Setelah menjalankan `pemodelan.py`, Anda akan mendapatkan output seperti:

```
🏆 MODEL TERBAIK: SVM
   Test F1 Macro Score: 0.8542
   Best Parameters: {'C': 10, 'kernel': 'linear', 'gamma': 'scale'}
```

File `model_comparison_results.csv` akan berisi:

| Model | Best_Params | CV_F1_Mean | Test_Accuracy | Test_F1_Macro | Test_ROC_AUC |
|-------|-------------|------------|---------------|---------------|--------------|
| SVM | {...} | 0.8421 | 0.8634 | 0.8542 | 0.9123 |
| Random Forest | {...} | 0.8312 | 0.8521 | 0.8401 | 0.8987 |
| Naive Bayes | {...} | 0.8156 | 0.8345 | 0.8234 | 0.8765 |

---

## 📝 Catatan Penting

### ✅ Kelebihan Penelitian Ini:
- Pipeline terstruktur dan sistematis
- Preprocessing komprehensif dengan kamus slang khusus
- Hyperparameter tuning dengan GridSearchCV
- Cross-validation untuk validasi model
- Handling imbalanced data dengan SMOTE
- Evaluasi multi-metrik

### ⚠️ Limitasi:
- Dataset terbatas pada TikTok dan YouTube
- Binary classification (tidak termasuk Netral)
- Kamus slang perlu update berkala
- Computational cost tinggi untuk GridSearchCV

---

## 👨‍💻 Author

**Tugas Akhir - Information Retrieval**  
STIS - Semester 5

---

## 📚 Referensi

1. Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
2. Joachims (1998) - "Text Categorization with Support Vector Machines"
3. Haddi et al. (2013) - "The Role of Text Pre-processing in Sentiment Analysis"
4. Sastrawi - Indonesian Stemmer Library

---

## 📞 Support

Jika ada pertanyaan atau issue, silakan buka issue di repository ini atau hubungi author.

---

**Last Updated**: November 2025
