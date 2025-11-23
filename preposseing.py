import pandas as pd
import re
import demoji
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# ==========================================
# 1. KONFIGURASI
# ==========================================
# Pastikan nama file ini sesuai dengan file label manual Anda di Colab
INPUT_FILE = 'data_labeling_tj_tiktokyt_final.csv' 
OUTPUT_FILE = 'data_siap_model_2class.csv'

# ==========================================
# 2. SETUP & KAMUS (UPDATED DARI DATA ANDA)
# ==========================================
print("[INIT] Menyiapkan Library & Kamus...")

try:
    demoji.download_codes()
except:
    pass 

# A. KAMUS SLANG (Diperbarui dari sample komentar Anda)
kamus_alay = {
    # Kata-kata dari sample Anda
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
    'wkwk': '', 'wkwkwk': '', 'haha': '', 'hihi': '' # Hapus tawa
}

# B. STOPWORD (HATI-HATI DENGAN NEGASI)
factory_stop = StopWordRemoverFactory()
stopwords_list = factory_stop.get_stop_words()

# PENTING: Kata ini JANGAN dihapus karena mengubah makna (Negatif/Positif)
kata_penting = ['tidak', 'bukan', 'jangan', 'tapi', 'belum', 'kurang', 'tapi']
stopwords_final = [word for word in stopwords_list if word not in kata_penting]

# Tambahan kata sampah (noise) yang tidak punya makna sentimen
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

# C. STEMMER
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

# ==========================================
# 3. FUNGSI PREPROCESSING (Updated)
# ==========================================

def clean_text_v2(text):
    """
    Membersihkan teks dari noise dan karakter yang tidak diperlukan.
    
    Args:
        text (str): Teks mentah yang akan dibersihkan
        
    Returns:
        str: Teks yang sudah dibersihkan
        
    Tahapan:
        1. Hapus emoji
        2. Konversi ke lowercase
        3. Normalisasi karakter berulang (contoh: "mahallll" -> "mahal")
        4. Hapus username, hashtag, URL, dan angka
        5. Hapus tanda baca dan simbol
        6. Hapus spasi berlebih
    """
    # 1. Hapus Emoji
    text = demoji.replace(text, '') 
    
    # 2. Lowercase
    text = str(text).lower()
    
    # 3. Hapus Karakter Berulang (Contoh: "mahallll" -> "mahal")
    # Ini penting untuk data sosmed
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 4. Hapus Username (@...), Hashtag (#...), URL, dan Angka
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text) # Hapus angka (5000, 7000) agar fokus ke kata sifat
    
    # 5. Hapus Tanda Baca & Simbol
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 6. Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_slang(text):
    """
    Mengubah kata-kata slang/alay menjadi kata baku menggunakan kamus.
    
    Args:
        text (str): Teks yang mengandung kata slang
        
    Returns:
        str: Teks dengan kata-kata yang sudah dinormalisasi
        
    Contoh:
        "tj naek bgt" -> "transjakarta naik banget"
    """
    words = text.split()
    new_text = []
    for word in words:
        if word in kamus_alay:
            new_text.append(kamus_alay[word])
        else:
            new_text.append(word)
    return " ".join(new_text)

def full_preprocessing(text):
    """
    Melakukan preprocessing lengkap pada teks komentar.
    
    Args:
        text (str): Teks komentar mentah
        
    Returns:
        str: Teks yang sudah dibersihkan dan dinormalisasi
        
    Tahapan:
        1. Cleaning (emoji, URL, mention, dll)
        2. Normalisasi slang
        3. Stopword removal (dengan mempertahankan kata negasi)
        4. Stemming (mengubah ke kata dasar)
        
    Contoh:
        Input:  "Wah TJ naek lagi 😭 mahalll bgt ga kuat bayarnya"
        Output: "transjakarta naik mahal tidak kuat bayar"
    """
    # Tahap 1: Cleaning Dasar
    text = clean_text_v2(text)
    
    # Tahap 2: Ganti Kata Alay
    text = normalize_slang(text)
    
    # Tahap 3: Stopword Removal
    if len(text) == 0:
        return ""
    text = stopword_remover.remove(text)
    
    # Tahap 4: Stemming (Mengubah ke kata dasar)
    # Contoh: "dinaikkan" -> "naik", "pelayanan" -> "layan"
    text = stemmer.stem(text)
    
    return text

# ==========================================
# 4. MAIN PROGRAM
# ==========================================
def main():
    print(f"\nMembaca file: {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Gagal baca file: {e}")
        return

    # Cek kolom komentar
    target_col = 'Komentar'
    if target_col not in df.columns:
        print("Kolom 'Komentar' tidak ditemukan! Pastikan format CSV benar.")
        return

    print(f"Total Data: {len(df)}")
    print("Mulai Preprocessing Ulang (Estimasi 5-10 Menit)...")
    
    # Pakai Swifter biar ngebut di Colab
    try:
        import swifter
        df['text_clean_baru'] = df[target_col].swifter.apply(full_preprocessing)
    except:
        print("Swifter tidak aktif, pakai mode biasa...")
        df['text_clean_baru'] = df[target_col].apply(full_preprocessing)
        
    # Bersihkan hasil yang kosong
    df['text_clean_baru'].replace('', float("NaN"), inplace=True)
    df.dropna(subset=['text_clean_baru'], inplace=True)
    
    # --- FINALISASI DATASET ---
    # Kita ambil kolom Komentar Asli, Text Clean Baru, dan LABEL YANG SUDAH ADA
    # Pastikan nama kolom label Anda di CSV sesuai (misal: Label_Teks atau Label_Angka)
    
    kolom_disimpan = ['Komentar', 'text_clean_baru']
    
    # Otomatis cari kolom label
    potential_labels = ['Label_Teks', 'Label_Angka', 'Label', 'Sentiment']
    for col in potential_labels:
        if col in df.columns:
            kolom_disimpan.append(col)
            print(f"-> Kolom Label '{col}' ditemukan dan akan disimpan.")
            
    df_final = df[kolom_disimpan]
    
    # Rename kolom biar standar
    df_final.rename(columns={'text_clean_baru': 'text_clean'}, inplace=True)
    
    # Simpan
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print("="*50)
    print(f"SUKSES! File siap modeling tersimpan di: {OUTPUT_FILE}")
    print("="*50)
    print(df_final.head(10))

if __name__ == "__main__":
    main()