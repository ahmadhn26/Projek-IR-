"""
Script untuk generate visualisasi EDA dan save sebagai images untuk web app.
Run script ini setelah training models.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import os

# Konfigurasi
INPUT_FILE = '../data_siap_model_2class.csv'
OUTPUT_DIR = 'static/images/eda'
col_text = 'text_clean'
col_label = 'Label_Teks'

def ensure_output_dir():
    """Create output directory if not exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ Created directory: {OUTPUT_DIR}")

def generate_visualizations():
    """Generate all EDA visualizations and save as images"""
    print("="*70)
    print("GENERATING EDA VISUALIZATIONS")
    print("="*70)
    
    ensure_output_dir()
    
    # Load data
    print("\n[1/6] Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"✓ Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Filter Netral if exists
    if 'Netral' in df[col_label].unique():
        df = df[df[col_label].isin(['Positif', 'Negatif'])].copy()
    
    # 1. Sentiment Distribution
    print("\n[2/6] Creating sentiment distribution chart...")
    plt.figure(figsize=(10, 6))
    order = df[col_label].value_counts().index.tolist()
    colors = ['#EF4444' if x == 'Negatif' else '#10B981' for x in order]
    sns.countplot(x=col_label, data=df, palette=colors, order=order)
    plt.title('Distribusi Sentimen', fontsize=16, fontweight='bold')
    plt.xlabel('Sentimen', fontsize=12)
    plt.ylabel('Jumlah Komentar', fontsize=12)
    
    # Add value labels on bars
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: sentiment_distribution.png")
    
    # 2. Word Cloud - Negative
    print("\n[3/6] Creating word cloud for negative sentiment...")
    neg_data = df[df[col_label] == 'Negatif'][col_text]
    if len(neg_data) > 0:
        text = ' '.join(neg_data.astype(str).tolist())
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            colormap='Reds',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Sentimen Negatif', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud_negative.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: wordcloud_negative.png")
    
    # 3. Word Cloud - Positive
    print("\n[4/6] Creating word cloud for positive sentiment...")
    pos_data = df[df[col_label] == 'Positif'][col_text]
    if len(pos_data) > 0:
        text = ' '.join(pos_data.astype(str).tolist())
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            colormap='Greens',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Sentimen Positif', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud_positive.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: wordcloud_positive.png")
    
    # 4. Text Length Distribution
    print("\n[5/6] Creating text length distribution...")
    df['text_length'] = df[col_text].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    df.boxplot(column='text_length', by=col_label, figsize=(12, 6))
    plt.suptitle('')  # Remove default title
    plt.title('Distribusi Panjang Teks per Sentimen', fontsize=16, fontweight='bold')
    plt.xlabel('Sentimen', fontsize=12)
    plt.ylabel('Jumlah Kata', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'text_length_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: text_length_distribution.png")
    
    # 5. Top Bigrams - Negative
    print("\n[6/6] Creating bigram analysis...")
    
    def get_top_ngrams(corpus, n=2, top_k=15):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    
    if len(neg_data) > 0:
        top_bigrams_neg = get_top_ngrams(neg_data.dropna(), n=2, top_k=15)
        x, y = zip(*top_bigrams_neg)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(y), y=list(x), palette='Reds_r')
        plt.title('Top 15 Bigram - Sentimen Negatif', fontsize=16, fontweight='bold')
        plt.xlabel('Frekuensi', fontsize=12)
        plt.ylabel('Bigram', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'bigrams_negative.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: bigrams_negative.png")
    
    # 6. Top Bigrams - Positive
    if len(pos_data) > 0:
        top_bigrams_pos = get_top_ngrams(pos_data.dropna(), n=2, top_k=15)
        x, y = zip(*top_bigrams_pos)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(y), y=list(x), palette='Greens_r')
        plt.title('Top 15 Bigram - Sentimen Positif', fontsize=16, fontweight='bold')
        plt.xlabel('Frekuensi', fontsize=12)
        plt.ylabel('Bigram', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'bigrams_positive.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: bigrams_positive.png")
    
    # Summary statistics
    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("="*70)
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. sentiment_distribution.png")
    print("  2. wordcloud_negative.png")
    print("  3. wordcloud_positive.png")
    print("  4. text_length_distribution.png")
    print("  5. bigrams_negative.png")
    print("  6. bigrams_positive.png")
    print("\n✓ Ready for web application!")

if __name__ == "__main__":
    generate_visualizations()
