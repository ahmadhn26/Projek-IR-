import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# --- KONFIGURASI ---
# Ganti nama file input ini jika file Anda berbeda
INPUT_FILE = 'data_siap_model_2class.csv' 
col_text = 'text_clean' 
col_label = 'Label_Teks' 

# ==========================================
# 1. VISUALISASI UTAMA (DISTRIBUSI & WORDCLOUD)
# ==========================================
def main_visualization():
    print(f"[INFO] Membaca data dari: {INPUT_FILE}...")
    try:
        # Load Data
        df = pd.read_csv(INPUT_FILE)
        print(f"Data berhasil dimuat: {len(df)} baris")
    except:
        print("ERROR: File tidak ditemukan! Pastikan file CSV di-upload dan nama file INPUT_FILE sudah benar.")
        return

    # Filter data jika masih ada Netral (Walaupun seharusnya sudah difilter di tahap sebelumnya)
    if 'Netral' in df[col_label].unique():
        print("[WARNING] Data Netral terdeteksi. Hanya akan memvisualisasikan Positif dan Negatif.")
        df = df[df[col_label].isin(['Positif', 'Negatif'])].copy()

    # --- A. BAR CHART DISTRIBUSI SENTIMEN ---
    plt.figure(figsize=(8, 5))
    # Order plot secara manual
    order = df[col_label].value_counts().index.tolist()
    sns.countplot(x=col_label, data=df, palette='viridis', order=order)
    plt.title('Distribusi Sentimen Transjakarta (Setelah Preprocessing)')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah Komentar')
    plt.show()

    print("\nTotal Sebaran Sentimen:")
    print(df[col_label].value_counts())

    # --- B. WORD CLOUD (Fokus Sentimen Negatif) ---
    def show_wordcloud(data, title):
        text = ' '.join(data.astype(str).tolist())
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color='white', 
            colormap='Reds_r', # Pakai warna merah/marun untuk Negatif
            max_words=100
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.show()

    # Word Cloud untuk Negatif
    neg_data = df[df[col_label] == 'Negatif'][col_text]
    if len(neg_data) > 0:
        print("\n--- WORDCLOUD SENTIMEN NEGATIF ---")
        show_wordcloud(neg_data, 'Kata Kunci Paling Sering di Sentimen NEGATIF')
    
    # Word Cloud untuk Positif
    pos_data = df[df[col_label] == 'Positif'][col_text]
    if len(pos_data) > 0:
        print("\n--- WORDCLOUD SENTIMEN POSITIF ---")
        # Ubah colormap untuk positif (hijau)
        def show_wordcloud_positive(data, title):
            text = ' '.join(data.astype(str).tolist())
            wordcloud = WordCloud(
                width=800, height=400, 
                background_color='white', 
                colormap='Greens',  # Warna hijau untuk Positif
                max_words=100
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16)
            plt.show()
        
        show_wordcloud_positive(pos_data, 'Kata Kunci Paling Sering di Sentimen POSITIF')

    # --- C. DISTRIBUSI PANJANG TEKS ---
    print("\n--- ANALISIS PANJANG TEKS ---")
    df['text_length'] = df[col_text].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(10, 5))
    df.boxplot(column='text_length', by=col_label, figsize=(8, 5))
    plt.title('Distribusi Panjang Teks per Sentimen')
    plt.suptitle('')  # Hapus judul default
    plt.ylabel('Jumlah Kata')
    plt.xlabel('Sentimen')
    plt.show()
    
    print("\nStatistik Panjang Teks:")
    print(df.groupby(col_label)['text_length'].describe())
    
    # --- D. BIGRAM ANALYSIS ---
    def get_top_ngrams(corpus, n=2, top_k=15):
        # Gunakan Bigram (2, 2)
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top_k]

    # Bigram untuk Negatif
    if len(neg_data) > 0:
        print("\n--- BIGRAM ANALYSIS: NEGATIF ---")
        top_bigrams_neg = get_top_ngrams(neg_data.dropna(), n=2, top_k=15)
        
        x, y = zip(*top_bigrams_neg)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(y), y=list(x), palette='Reds_r')
        plt.title('Top 15 Frase (Bigram) Paling Sering Muncul di Sentimen Negatif')
        plt.xlabel('Frekuensi')
        plt.ylabel('Frase (Bigram)')
        plt.tight_layout()
        plt.show()
    
    # Bigram untuk Positif
    if len(pos_data) > 0:
        print("\n--- BIGRAM ANALYSIS: POSITIF ---")
        top_bigrams_pos = get_top_ngrams(pos_data.dropna(), n=2, top_k=15)
        
        x, y = zip(*top_bigrams_pos)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(y), y=list(x), palette='Greens_r')
        plt.title('Top 15 Frase (Bigram) Paling Sering Muncul di Sentimen Positif')
        plt.xlabel('Frekuensi')
        plt.ylabel('Frase (Bigram)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main_visualization()