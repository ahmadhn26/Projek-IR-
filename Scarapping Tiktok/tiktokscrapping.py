import time
import random
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==========================================
# 1. KONFIGURASI
# ==========================================

VIDEO_URLS = [
    "https://www.tiktok.com/@rozak.idx/photo/7565037699082554631?q=tarif%20transjakarta&t=1763737147719",
    "https://www.tiktok.com/@liputan6.sctv/video/7565790379098836245?q=tarif%20transjakarta&t=1763737147719"
]

MAX_COMMENTS_PER_VIDEO = 500 
FILE_OUTPUT = "data_tiktok_transjakarta_final.csv"

# ==========================================
# 2. PROGRAM UTAMA
# ==========================================

def main():
    print("[INFO] Menyiapkan Browser Anti-Deteksi...")
    
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    
    driver = uc.Chrome(options=options)
    all_data = []

    try:
        # --- FASE 1: LOGIN MANUAL ---
        driver.get("https://www.tiktok.com/login")
        
        print("\n" + "="*60)
        print("!!! SILAKAN LOGIN MANUAL !!!")
        print("1. Login akun TikTok.")
        print("2. Pastikan masuk sampai BERANDA.")
        print("="*60)
        
        input(">>> TEKAN ENTER DI SINI JIKA SUDAH BERHASIL LOGIN...")
        
        # --- FASE 2: LOOPING VIDEO ---
        for index, url in enumerate(VIDEO_URLS):
            print(f"\n[{index+1}/{len(VIDEO_URLS)}] Membuka video: {url}")
            driver.get(url)
            time.sleep(random.uniform(5, 8)) 
            
            # Scroll awal
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(2)

            current_video_comments = set()
            
            # Variabel untuk mendeteksi stuck/habis
            stuck_counter = 0 
            
            print(f"   Mulai mengambil komentar (Target: {MAX_COMMENTS_PER_VIDEO})...")
            
            while len(current_video_comments) < MAX_COMMENTS_PER_VIDEO:
                
                # Simpan jumlah komentar SEBELUM scraping ulang
                prev_count = len(current_video_comments)

                # --- AMBIL ELEMEN ---
                comment_elements = []
                try:
                    # Coba berbagai selector
                    comment_elements = driver.find_elements(By.XPATH, '//span[@data-e2e="comment-level-1"]')
                    if not comment_elements:
                        comment_elements = driver.find_elements(By.XPATH, '//div[contains(@class, "DivCommentContentContainer")]')
                except:
                    pass

                # --- EKSTRAK TEXT ---
                for el in comment_elements:
                    try:
                        text = el.text.replace('\n', ' ').strip()
                        if len(text) > 1 and text not in current_video_comments:
                            current_video_comments.add(text)
                            all_data.append([text, url])
                    except:
                        continue
                
                # Hitung jumlah komentar SETELAH scraping
                curr_count = len(current_video_comments)
                
                # Tampilkan status
                print(f"\r   -> Terkumpul: {curr_count} komentar...", end="", flush=True)

                # --- LOGIKA STOP JIKA KOMENTAR HABIS ---
                if curr_count == prev_count:
                    # Jika jumlahnya TIDAK BERUBAH setelah scroll
                    stuck_counter += 1
                    # Jika sudah 5 kali scroll tapi tidak ada data baru, berarti HABIS
                    if stuck_counter >= 5:
                        print("\n   [STOP] Komentar sudah habis atau mentok.")
                        break
                else:
                    # Jika ada data baru, reset counter
                    stuck_counter = 0

                if curr_count >= MAX_COMMENTS_PER_VIDEO:
                    print("\n   [STOP] Target tercapai.")
                    break

                # --- SCROLLING ---
                driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(random.uniform(2, 3)) # Waktu tunggu loading komentar baru
            
            print(f"\n   Selesai video ini. Total: {len(current_video_comments)}")
            
            # Jeda Manual
            if index < len(VIDEO_URLS) - 1:
                print("-" * 40)
                input(">>> TEKAN ENTER UNTUK LANJUT KE VIDEO BERIKUTNYA...")
                print("-" * 40)

    except KeyboardInterrupt:
        print("\n\n[STOP] Dihentikan pengguna.")

    except Exception as e:
        print(f"\n\n[ERROR] {e}")

    finally:
        if all_data:
            df = pd.DataFrame(all_data, columns=['Komentar', 'Sumber_Video'])
            df.to_csv(FILE_OUTPUT, index=False)
            print(f"\n[SUKSES] {len(df)} data tersimpan di {FILE_OUTPUT}")
        else:
            print("\n[ZONK] Tidak ada data.")
            
        driver.quit()

if __name__ == "__main__":
    main()