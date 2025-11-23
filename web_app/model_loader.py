"""
Model loader dan preprocessing utilities untuk web application.
"""

import os
import re
import joblib
import demoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Konfigurasi
MODELS_DIR = 'models'

# Setup preprocessing tools
try:
    demoji.download_codes()
except:
    pass

# Kamus slang (sama dengan preposseing.py)
KAMUS_ALAY = {
    'abot': 'berat', 'adain': 'adakan', 'naek': 'naik', 'kaga': 'tidak', 'ga': 'tidak',
    'transum': 'transportasi umum', 'tj': 'transjakarta', 'teje': 'transjakarta',
    'jeklingko': 'mikrotrans', 'jaklingko': 'mikrotrans', 'angkot': 'mikrotrans',
    'ngadi': 'mengada', 'kudet': 'kurang update', 'songong': 'sombong',
    'bener': 'benar', 'dikit': 'sedikit', 'banyak': 'banyak', 'smpe': 'sampai',
    'karna': 'karena', 'krn': 'karena', 'utk': 'untuk', 'yg': 'yang', 'dg': 'dengan',
    'gmn': 'bagaimana', 'blm': 'belum', 'udh': 'sudah', 'sdh': 'sudah',
    'bgt': 'banget', 'kalo': 'kalau', 'kl': 'kalau', 'klo': 'kalau',
    'tp': 'tapi', 'tpi': 'tapi', 'sy': 'saya', 'gw': 'saya', 'gua': 'saya', 'aku': 'saya',
    'lo': 'kamu', 'lu': 'kamu', 'emang': 'memang', 'kek': 'seperti',
    'dl': 'dulu', 'dlu': 'dulu', 'aj': 'saja', 'aja': 'saja',
    'msh': 'masih', 'brp': 'berapa', 'bs': 'bisa', 'bisa2': 'bisa',
    'ortu': 'orang tua', 'bapa': 'bapak', 'bpk': 'bapak', 'gub': 'gubernur',
    'dpt': 'dapat', 'liat': 'lihat', 'ngomongin': 'bicara',
    'pp': 'pulang pergi', 'tap': 'tempel', 'saldo': 'uang',
    'goceng': 'lima ribu', 'ceban': 'sepuluh ribu', '5k': 'lima ribu', '7k': 'tujuh ribu', '10k': 'sepuluh ribu',
    'njs': 'najis', 'betmut': 'bad mood', 'anjing': 'anjing', 'anjir': 'anjing',
    'wkwk': '', 'wkwkwk': '', 'haha': '', 'hihi': ''
}

# Stopwords
factory_stop = StopWordRemoverFactory()
stopwords_list = factory_stop.get_stop_words()
kata_penting = ['tidak', 'bukan', 'jangan', 'tapi', 'belum', 'kurang', 'tapi']
stopwords_final = [word for word in stopwords_list if word not in kata_penting]
stopwords_final.extend([
    'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'kalo', 'klo', 
    'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 
    'nih', 'sih', 'si', 'tau', 'tuh', 'utk', 'ya', 
    'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
    'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', '&amp', 'yah',
    'banget', 'dong', 'kok', 'mah', 'deh', 'kan', 'gan', 'sis', 'bro',
    'kak', 'min', 'pak', 'bu', 'mas', 'mbak', 'om', 'tante', 'reply'
])

dictionary_stop = ArrayDictionary(stopwords_final)
stopword_remover = StopWordRemover(dictionary_stop)

# Stemmer
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()


def clean_text(text):
    """Clean text dari noise"""
    text = demoji.replace(text, '')
    text = str(text).lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_slang(text):
    """Normalize kata slang"""
    words = text.split()
    new_text = []
    for word in words:
        if word in KAMUS_ALAY:
            new_text.append(KAMUS_ALAY[word])
        else:
            new_text.append(word)
    return " ".join(new_text)


def preprocess_text(text):
    """
    Full preprocessing pipeline untuk input text.
    Returns: (cleaned_text, preprocessing_steps)
    """
    steps = []
    
    # Original
    steps.append(('Original', text))
    
    # Step 1: Clean
    text = clean_text(text)
    steps.append(('Cleaned', text))
    
    # Step 2: Normalize slang
    text = normalize_slang(text)
    steps.append(('Normalized', text))
    
    # Step 3: Stopword removal
    if len(text) > 0:
        text = stopword_remover.remove(text)
    steps.append(('Stopwords Removed', text))
    
    # Step 4: Stemming
    text = stemmer.stem(text)
    steps.append(('Stemmed', text))
    
    return text, steps


class ModelLoader:
    """Class untuk load dan manage models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.model_results = None
        self.dataset_stats = None
        
    def load_all(self):
        """Load semua models dan data"""
        print("Loading models...")
        
        # Load TF-IDF vectorizer
        vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
            print("✓ TF-IDF vectorizer loaded")
        else:
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}")
        
        # Load models
        model_files = {
            'Naive Bayes': 'naive_bayes_model.pkl',
            'SVM': 'svm_model.pkl',
            'Random Forest': 'random_forest_model.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"✓ {name} model loaded")
            else:
                print(f"⚠ {name} model not found at {model_path}")
        
        # Load results
        import pandas as pd
        results_path = os.path.join(MODELS_DIR, 'model_results.csv')
        if os.path.exists(results_path):
            self.model_results = pd.read_csv(results_path)
            print("✓ Model results loaded")
        
        # Load dataset stats
        import json
        stats_path = os.path.join(MODELS_DIR, 'dataset_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.dataset_stats = json.load(f)
            print("✓ Dataset stats loaded")
        
        print(f"✓ Loaded {len(self.models)} models successfully\n")
        
    def predict(self, text, model_name='SVM'):
        """
        Predict sentiment untuk input text.
        
        Args:
            text (str): Input text
            model_name (str): Model name ('Naive Bayes', 'SVM', 'Random Forest')
            
        Returns:
            dict: {
                'sentiment': 'Positif' or 'Negatif',
                'confidence': float,
                'probabilities': {'Negatif': float, 'Positif': float},
                'preprocessed_text': str,
                'preprocessing_steps': list
            }
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        # Preprocess
        preprocessed_text, steps = preprocess_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([preprocessed_text])
        
        # Predict
        model = self.models[model_name]
        prediction = model.predict(X)[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            probabilities = {
                'Negatif': float(proba[0]),
                'Positif': float(proba[1])
            }
            confidence = float(max(proba))
        else:
            # For models without predict_proba
            probabilities = None
            confidence = 1.0
        
        sentiment = 'Positif' if prediction == 1 else 'Negatif'
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probabilities,
            'preprocessed_text': preprocessed_text,
            'preprocessing_steps': steps
        }
    
    def get_model_info(self, model_name):
        """Get info tentang specific model"""
        if self.model_results is not None:
            model_data = self.model_results[self.model_results['Model'] == model_name]
            if not model_data.empty:
                return model_data.iloc[0].to_dict()
        return None
    
    def get_all_models_info(self):
        """Get info semua models"""
        if self.model_results is not None:
            return self.model_results.to_dict('records')
        return []
    
    def get_dataset_stats(self):
        """Get dataset statistics"""
        return self.dataset_stats if self.dataset_stats else {}


# Global model loader instance
model_loader = ModelLoader()
