

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings untuk output yang lebih bersih

# --- KONFIGURASI ---
# Ganti dengan nama file binary classification Anda yang baru
INPUT_FILE = 'data_siap_model_2class.csv' 
TARGET_NAMES = ['Negatif', 'Positif'] # Hanya 2 kelas target

def main():
    """
    Fungsi utama untuk melakukan pemodelan sentiment analysis dengan:
    - Hyperparameter tuning menggunakan GridSearchCV
    - Cross-validation untuk validasi model
    - Evaluasi komprehensif dengan multiple metrics
    """
    # 1. Load Data
    print("Membaca data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("File tidak ditemukan! Upload dulu file CSV hasil labeling.")
        return

    # Pastikan tidak ada nilai kosong di text
    df = df.dropna(subset=['text_clean', 'Label_Angka'])
    
    # --- Filter Data: HAPUS NETRAL (0) jika ada ---
    df = df[df['Label_Angka'] != 0].copy()
    
    # --- Mapping Label Angka: Negatif (-1) menjadi 0, Positif (1) tetap 1 ---
    # Ini penting agar SVM/NB lebih stabil dalam binary classification
    df['Label_Angka'] = df['Label_Angka'].replace(-1, 0)
    
    X_text = df['text_clean']
    y = df['Label_Angka'] # y sekarang hanya berisi 0 dan 1

    print(f"Total Data Final (Binary): {len(df)}")
    print(f"Sebaran Awal: {Counter(y)}")

    # ---------------------------------------------------------
    # TAHAP 1: VEKTORISASI (TF-IDF + Bigram)
    # ---------------------------------------------------------
    print("\nMengubah Teks menjadi Angka (TF-IDF Bigram)...")
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,  # Abaikan kata yang muncul < 2 kali (mengurangi noise)
        max_df=0.8  # Abaikan kata yang muncul di >80% dokumen (terlalu umum)
    )
    X_tfidf = tfidf.fit_transform(X_text)

    # ---------------------------------------------------------
    # TAHAP 2: SPLIT TRAIN-TEST (STRATIFIED)
    # ---------------------------------------------------------
    print("Membagi Data (80% Train : 20% Test) dengan Stratified...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # TAHAP 3: PENYEIMBANGAN DATA (SMOTE)
    # ---------------------------------------------------------
    print("Menerapkan SMOTE pada Data Training...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Data Train Setelah SMOTE: {X_train_resampled.shape[0]} (Seimbang)")

    # ---------------------------------------------------------
    # TAHAP 4: HYPERPARAMETER TUNING & CROSS-VALIDATION
    # ---------------------------------------------------------
    
    # Definisikan parameter grid untuk setiap model
    param_grids = {
        'Naive Bayes': {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    base_models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(probability=True, random_state=42),  # probability=True untuk ROC-AUC
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    results = []
    best_models = {}

    print("\n" + "="*60)
    print("MULAI HYPERPARAMETER TUNING & TRAINING (BINARY CLASSIFICATION)")
    print("="*60)

    for name in base_models.keys():
        print(f"\n{'='*60}")
        print(f"MODEL: {name}")
        print(f"{'='*60}")
        
        # ---------------------------------------------------------
        # TAHAP 4A: GRID SEARCH untuk mencari parameter terbaik
        # ---------------------------------------------------------
        print(f"\n[1/3] Melakukan Grid Search untuk {name}...")
        grid_search = GridSearchCV(
            estimator=base_models[name],
            param_grid=param_grids[name],
            cv=5,  # 5-fold cross-validation
            scoring='f1_macro',
            n_jobs=-1,  # Gunakan semua CPU cores
            verbose=1
        )
        
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        print(f"\n✓ Parameter Terbaik untuk {name}:")
        print(f"  {grid_search.best_params_}")
        print(f"  Best CV F1 Score: {grid_search.best_score_:.4f}")
        
        # ---------------------------------------------------------
        # TAHAP 4B: CROSS-VALIDATION pada model terbaik
        # ---------------------------------------------------------
        print(f"\n[2/3] Melakukan 5-Fold Cross-Validation...")
        cv_scores = cross_val_score(
            best_model, 
            X_train_resampled, 
            y_train_resampled, 
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        print(f"  CV F1 Scores: {cv_scores}")
        print(f"  Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # ---------------------------------------------------------
        # TAHAP 4C: EVALUASI pada Test Set
        # ---------------------------------------------------------
        print(f"\n[3/3] Evaluasi pada Test Set...")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # Hitung Metrik Evaluasi
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC Score (jika model support probability)
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        results.append({
            'Model': name,
            'Best_Params': str(grid_search.best_params_),
            'CV_F1_Mean': cv_scores.mean(),
            'CV_F1_Std': cv_scores.std(),
            'Test_Accuracy': acc,
            'Test_F1_Macro': f1_macro,
            'Test_F1_Weighted': f1_weighted,
            'Test_ROC_AUC': roc_auc
        })
        
        print(f"\n--- Hasil {name} pada Test Set ---")
        print(f"  Akurasi: {acc*100:.2f}%")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        if roc_auc:
            print(f"  ROC-AUC: {roc_auc:.4f}")
        
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))
        
        # Visualisasi Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=TARGET_NAMES, 
                    yticklabels=TARGET_NAMES,
                    cbar_kws={'label': 'Jumlah Prediksi'})
        plt.title(f'Confusion Matrix - {name}\n(Tuned Parameters)', fontsize=12, fontweight='bold')
        plt.ylabel('Label Asli')
        plt.xlabel('Prediksi Model')
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # TAHAP 5: KESIMPULAN PERBANDINGAN
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("PERBANDINGAN AKHIR - SEMUA MODEL")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    
    # Tampilkan tabel lengkap
    print("\nTabel Lengkap Hasil Evaluasi:")
    print(results_df.to_string(index=False))
    
    # Simpan hasil ke CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n✓ Hasil perbandingan disimpan ke: model_comparison_results.csv")
    
    # Tentukan model terbaik berdasarkan Test F1 Macro
    results_df_sorted = results_df.sort_values(by='Test_F1_Macro', ascending=False)
    best_model_name = results_df_sorted.iloc[0]['Model']
    best_f1 = results_df_sorted.iloc[0]['Test_F1_Macro']
    
    print("\n" + "="*70)
    print(f"🏆 MODEL TERBAIK: {best_model_name}")
    print(f"   Test F1 Macro Score: {best_f1:.4f}")
    print(f"   Best Parameters: {results_df_sorted.iloc[0]['Best_Params']}")
    print("="*70)
    
    # Visualisasi Perbandingan Model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Perbandingan F1 Scores
    ax1 = axes[0]
    x_pos = range(len(results_df))
    ax1.bar(x_pos, results_df['CV_F1_Mean'], alpha=0.7, label='CV F1 (Mean)', color='skyblue')
    ax1.bar([x + 0.3 for x in x_pos], results_df['Test_F1_Macro'], alpha=0.7, label='Test F1 Macro', color='orange')
    ax1.set_xticks([x + 0.15 for x in x_pos])
    ax1.set_xticklabels(results_df['Model'])
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Perbandingan F1 Score: CV vs Test')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Perbandingan Semua Metrik
    ax2 = axes[1]
    metrics_to_plot = results_df[['Test_Accuracy', 'Test_F1_Macro', 'Test_F1_Weighted']].T
    metrics_to_plot.columns = results_df['Model']
    metrics_to_plot.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_ylabel('Score')
    ax2.set_title('Perbandingan Metrik Evaluasi pada Test Set')
    ax2.set_xticklabels(['Accuracy', 'F1 Macro', 'F1 Weighted'], rotation=45)
    ax2.legend(title='Model')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight')
    print("\n✓ Grafik perbandingan disimpan ke: model_comparison_chart.png")
    plt.show()
    
    print("\n" + "="*70)
    print("SELESAI! Semua model telah dilatih dan dievaluasi dengan optimal.")
    print("="*70)

if __name__ == "__main__":
    main()