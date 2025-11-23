"""
Script untuk training model dan menyimpan hasil untuk web application.
Modifikasi dari pemodelan.py dengan tambahan save model functionality.
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path untuk import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Konfigurasi
INPUT_FILE = '../data_siap_model_2class.csv'
MODELS_DIR = 'models'
TARGET_NAMES = ['Negatif', 'Positif']

def ensure_models_dir():
    """Create models directory if not exists"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"✓ Created directory: {MODELS_DIR}")

def train_and_save_models():
    """
    Train models dengan hyperparameter tuning dan save hasilnya.
    """
    print("="*70)
    print("TRAINING & SAVING MODELS FOR WEB APPLICATION")
    print("="*70)
    
    ensure_models_dir()
    
    # 1. Load Data
    print("\n[1/5] Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"✓ Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Preprocessing
    df = df.dropna(subset=['text_clean', 'Label_Angka'])
    df = df[df['Label_Angka'] != 0].copy()
    df['Label_Angka'] = df['Label_Angka'].replace(-1, 0)
    
    X_text = df['text_clean']
    y = df['Label_Angka']
    
    print(f"  Total samples: {len(df)}")
    print(f"  Distribution: {Counter(y)}")
    
    # 2. TF-IDF Vectorization
    print("\n[2/5] TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X_tfidf = tfidf.fit_transform(X_text)
    print(f"✓ TF-IDF shape: {X_tfidf.shape}")
    
    # Save TF-IDF vectorizer
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    print(f"✓ Saved: tfidf_vectorizer.pkl")
    
    # 3. Train-Test Split
    print("\n[3/5] Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"✓ Training samples after SMOTE: {X_train_resampled.shape[0]}")
    
    # 4. Model Training dengan Hyperparameter Tuning
    print("\n[4/5] Training models with hyperparameter tuning...")
    
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
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = []
    best_models = {}
    
    for name in base_models.keys():
        print(f"\n  Training {name}...")
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_models[name],
            param_grid=param_grids[name],
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        results.append({
            'Model': name,
            'Best_Params': str(grid_search.best_params_),
            'CV_F1_Mean': grid_search.best_score_,
            'Test_Accuracy': acc,
            'Test_F1_Macro': f1_macro,
            'Test_F1_Weighted': f1_weighted,
            'Test_ROC_AUC': roc_auc
        })
        
        print(f"  ✓ {name}: Accuracy={acc:.4f}, F1={f1_macro:.4f}")
        
        # Save model
        model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(best_model, os.path.join(MODELS_DIR, model_filename))
        print(f"  ✓ Saved: {model_filename}")
    
    # 5. Save Results
    print("\n[5/5] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_results.csv'), index=False)
    print(f"✓ Saved: model_results.csv")
    
    # Save dataset statistics
    stats = {
        'total_samples': len(df),
        'negative_samples': int((y == 0).sum()),
        'positive_samples': int((y == 1).sum()),
        'test_samples': len(y_test),
        'train_samples': len(y_train_resampled)
    }
    
    import json
    with open(os.path.join(MODELS_DIR, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: dataset_stats.json")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    
    best_model_name = results_df.sort_values('Test_F1_Macro', ascending=False).iloc[0]['Model']
    print(f"\n🏆 Best Model: {best_model_name}")
    print("\n✓ All models and data saved to 'models/' directory")
    print("✓ Ready for web application!")

if __name__ == "__main__":
    train_and_save_models()
