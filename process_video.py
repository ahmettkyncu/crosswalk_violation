import cv2
import os
import shutil
from ultralytics import YOLO
from shapely.geometry import box as shapely_box, Point
import requests
import io
import argparse
import time
import math
import numpy as np 
import re 
from collections import Counter 
import easyocr

# --- AYARLAR ---
DEBUG_KLASORU = "debug_plates"
os.makedirs(DEBUG_KLASORU, exist_ok=True)

# 1. ARGUMANLAR
arguman_ayristirici = argparse.ArgumentParser()
arguman_ayristirici.add_argument("--video", required=True, help="Video yolu")
arguman_ayristirici.add_argument("--camera", required=True, help="Kamera adi")
argumanlar = arguman_ayristirici.parse_args()

video_yolu = argumanlar.video
MEVCUT_KAMERA_ADI = argumanlar.camera
SUNUCU_API_ADRESI = "http://127.0.0.1:2000/api/upload"

# 2. MODELLER
print(f"[{MEVCUT_KAMERA_ADI}] Nesne Tespit Modelleri yukleniyor...")
yaya_gecidi_modeli = YOLO("weights/crosswalk_best.pt")
yaya_modeli = YOLO("weights/person_best.pt")
arac_modeli = YOLO("weights/vehicle_best.pt")
trafik_isigi_modeli = YOLO("weights/trafficlight_best.pt")
plaka_modeli = YOLO("weights/plate_best.pt")

print(f"[{MEVCUT_KAMERA_ADI}] 📖 OCR Motoru (EasyOCR) başlatılıyor...")
reader = easyocr.Reader(['en'], gpu=True) 
print(f"[{MEVCUT_KAMERA_ADI}] 📖 EasyOCR Hazır.\n")

# ==============================================================================
# YENİ EKLENEN FONKSİYON: TÜRKİYE PLAKA KONTROLÜ (REGEX)
# ==============================================================================
def is_valid_turkish_plate(text):
    pattern = r"^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}\d{2,4}$"
    return re.match(pattern, text) is not None

def preprocess_plate(img):
    if img is None or img.size == 0: return None
    scale = 3 
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    thresh_temp = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh_temp > 0))
    angle = 0
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
    if abs(angle) > 1 and abs(angle) < 45:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed = clahe.apply(gray)
    return processed
# ==============================================================================

# 3. VIDEO
video_kaynagi = cv2.VideoCapture(video_yolu)
if not video_kaynagi.isOpened():
    print(f"[{MEVCUT_KAMERA_ADI}] ⚠️ Video acilamadi:", video_yolu)
    exit()

kare_sayaci = 0
print(f"[{MEVCUT_KAMERA_ADI}] 🎥 Video isleniyor... (Q ile cikabilirsin)\n")

# --- PARAMETRELER ---
KARE_ATLAMA = 1 
IC_BOSLUK = 47 
IHLAL_BEKLEME_SURESI = 5.0 

# >>>>> YENİ AYAR: Veritabanına gönderme gecikmesi (Saniye) <<<<<
VERITABANI_GONDERIM_GECIKMESI = 5.0 

# --- HAFIZA ---
sabit_dis_kutu = None
sabit_ic_guvenli_alan = None
sabit_dis_alan = None      
son_islenen_kare = None
son_ihlaller_listesi = [] 

gonderilen_araclar_db = {}      
gonderilen_plakali_araclar = set() 
arac_plaka_havuzu = {}       
arac_plaka_son_karar = {}    
onceki_karedeki_idler = set() 

# >>>>> YENİ HAFIZA: Bekleyen Yüklemeler Kuyruğu <<<<<
# Format: { track_id: { 'data': dict, 'image': img, 'first_seen': timestamp } }
bekleyen_yuklemeler = {} 

while True:
    basarili, kare = video_kaynagi.read()
    if not basarili: break
    kare_sayaci += 1

    simdiki_zaman = time.time()
    son_ihlaller_listesi = [x for x in son_ihlaller_listesi if simdiki_zaman - x['time'] < IHLAL_BEKLEME_SURESI]

    # ROI TESPITI
    if kare_sayaci % 5 == 0 and sabit_ic_guvenli_alan is None:
        gecit_sonuclari = yaya_gecidi_modeli(kare, verbose=False)
        if gecit_sonuclari[0].boxes:
            ham_kutu = gecit_sonuclari[0].boxes.xyxy.cpu().numpy()[0]
            gx1, gy1, gx2, gy2 = map(int, ham_kutu)
            
            ic_gx1 = gx1 + IC_BOSLUK
            ic_gy1 = gy1 + IC_BOSLUK
            ic_gx2 = gx2 - IC_BOSLUK
            ic_gy2 = gy2 - IC_BOSLUK
            
            if not (ic_gx1 >= ic_gx2 or ic_gy1 >= ic_gy2):
                sabit_dis_kutu = [gx1, gy1, gx2, gy2]
                sabit_ic_kutu_koordinatlari = [ic_gx1, ic_gy1, ic_gx2, ic_gy2]
                sabit_ic_guvenli_alan = shapely_box(ic_gx1, ic_gy1, ic_gx2, ic_gy2) 
                sabit_dis_alan = shapely_box(gx1, gy1, gx2, gy2)           
                print(f"[{MEVCUT_KAMERA_ADI}] ✅ Yaya yolu ROI sabitlendi.")

    # ANA ISLEME
    if kare_sayaci % KARE_ATLAMA == 0 and sabit_ic_guvenli_alan is not None:
        
        yaya_sonuclari = yaya_modeli(kare, verbose=False)
        arac_sonuclari = arac_modeli.track(kare, persist=True, verbose=False)
        isik_sonuclari = trafik_isigi_modeli(kare, verbose=False)
        
        yaya_kutulari = yaya_sonuclari[0].boxes.xyxy.cpu().numpy() if yaya_sonuclari[0].boxes else []
        isik_kutulari_objesi = isik_sonuclari[0].boxes if isik_sonuclari[0].boxes else []

        # Trafik Işığı
        yaya_gecidi_guvenli = True
        gecit_rengi = (0, 255, 0)
        durum_metni = ""
        ihlal_sebebi = None
        trafik_isigi_durumu = None

        for isik_objesi in isik_kutulari_objesi:
            sinif_id = int(isik_objesi.cls[0])
            if sinif_id == 0: trafik_isigi_durumu = "yesil"
            elif sinif_id == 1: trafik_isigi_durumu = "kirmizi"

        yaya_gecitte_var = any(
            sabit_dis_alan.intersects(shapely_box(int(yx1), int(yy1), int(yx2), int(yy2)))
            for (yx1, yy1, yx2, yy2) in yaya_kutulari
        )

        arac_kutulari_track = arac_sonuclari[0].boxes.xyxy.cpu().numpy() if arac_sonuclari[0].boxes else []

        if trafik_isigi_durumu == "yesil":
            durum_metni = "Yaya Geciyor (Arac Durmali)"
            gecit_rengi = (0, 0, 255) 
            if any(sabit_ic_guvenli_alan.contains(Point(int((box[0]+box[2])/2), int(box[3]))) for box in arac_kutulari_track):
                yaya_gecidi_guvenli = False
                ihlal_sebebi = "Arac yaya isigi yesilken gecidi isgal etti"
        elif trafik_isigi_durumu == "kirmizi":
            durum_metni = "Arac Geciyor"
            gecit_rengi = (0, 255, 0) 
            yaya_gecidi_guvenli = True 
        else:
            durum_metni = "Kontrolsuz Gecit"
            gecit_rengi = (255, 255, 0) 
            if yaya_gecitte_var and any(sabit_ic_guvenli_alan.contains(Point(int((box[0]+box[2])/2), int(box[3]))) for box in arac_kutulari_track):
                yaya_gecidi_guvenli = False
                ihlal_sebebi = "Arac isiksiz yolda yayaya yol vermedi"

        # Çizimler
        gx1, gy1, gx2, gy2 = sabit_dis_kutu
        cv2.rectangle(kare, (gx1, gy1), (gx2, gy2), gecit_rengi, 3)
        ic_gx1, ic_gy1, ic_gx2, ic_gy2 = sabit_ic_kutu_koordinatlari
        cv2.rectangle(kare, (ic_gx1, ic_gy1), (ic_gx2, ic_gy2), (255, 255, 255), 2)
        
        # --- ARAÇ TAKİP VE PLAKA İŞLEMLERİ ---
        mevcut_karedeki_idler = set()

        if arac_sonuclari[0].boxes is not None and arac_sonuclari[0].boxes.id is not None:
            track_kutular = arac_sonuclari[0].boxes.xyxy.cpu().numpy()
            track_idler = arac_sonuclari[0].boxes.id.int().cpu().tolist()

            for kutu, track_id in zip(track_kutular, track_idler):
                mevcut_karedeki_idler.add(track_id)
                ax1, ay1, ax2, ay2 = map(int, kutu)
                
                # Aracı Çiz
                cv2.rectangle(kare, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
                mevcut_plaka = arac_plaka_son_karar.get(track_id, "Okunuyor...")
                cv2.putText(kare, mevcut_plaka, (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                on_nokta_x = int((ax1 + ax2) / 2)
                on_nokta_y = int(ay2)
                arac_gecitte_mi = sabit_ic_guvenli_alan.contains(Point(on_nokta_x, on_nokta_y))

                if arac_gecitte_mi:
                    if (ax2 - ax1) > 40: 
                        arac_kirpma = kare[ay1:ay2, ax1:ax2]
                        
                        # --- EASYOCR OCR ---
                        try:
                            # 1. Detection
                            plaka_sonuclari = plaka_modeli(arac_kirpma, verbose=False)
                            if plaka_sonuclari[0].boxes:
                                kutular = plaka_sonuclari[0].boxes.xyxy.cpu().numpy()
                                en_iyi_kutu = max(kutular, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                                plx1, ply1, plx2, ply2 = map(int, en_iyi_kutu)
                                
                                h_img, w_img = arac_kirpma.shape[:2]
                                pad = 20 
                                plx1 = max(0, plx1 - pad); ply1 = max(0, ply1 - pad)
                                plx2 = min(w_img, plx2 + pad); ply2 = min(h_img, ply2 + pad)
                                plaka_kirpma = arac_kirpma[ply1:ply2, plx1:plx2]

                                # 2. Preprocess
                                islenmis_plaka = preprocess_plate(plaka_kirpma)

                                if islenmis_plaka is not None:
                                    # 3. EasyOCR
                                    ocr_sonuclari = reader.readtext(
                                        islenmis_plaka, 
                                        detail=0, 
                                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                                    )
                                    ham_metin = "".join(ocr_sonuclari)
                                    temiz_metin = re.sub(r'[^A-Z0-9]', '', ham_metin.upper().strip())

                                    # --- TÜRKİYE PLAKA KONTROLÜ ---
                                    if is_valid_turkish_plate(temiz_metin):
                                        if track_id not in arac_plaka_havuzu:
                                            arac_plaka_havuzu[track_id] = []
                                        arac_plaka_havuzu[track_id].append(temiz_metin)
                                        
                                        en_cok_gecen = Counter(arac_plaka_havuzu[track_id]).most_common(1)[0][0]
                                        arac_plaka_son_karar[track_id] = en_cok_gecen
                                    else:
                                        pass

                        except Exception as e:
                             pass

                # --- GÖNDERİM KONTROLÜ (GÜNCELLENDİ: KUYRUK SİSTEMİ) ---
                if arac_gecitte_mi:
                    arac_status = "normal"
                    arac_reason = "Gecis Yapti"
                    durum_rengi = (0, 255, 0)

                    if not yaya_gecidi_guvenli: 
                        arac_status = "violation"
                        arac_reason = ihlal_sebebi if ihlal_sebebi else "Kural Ihlali"
                        durum_rengi = (0, 0, 255)

                    cv2.putText(kare, f"ID:{track_id} {arac_status}", (ax1, ay1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, durum_rengi, 2)

                    if track_id not in gonderilen_araclar_db:
                        gonderilen_araclar_db[track_id] = set()

                    final_plaka = arac_plaka_son_karar.get(track_id, "OKUNAMADI")
                    plaka_okundu = (final_plaka != "OKUNAMADI")
                    durum_anahtari = f"{track_id}_{arac_reason}"
                    
                    kuyruaga_ekle = False

                    # 1. Hiç gönderilmemişse -> Kuyruğa Ekle
                    if arac_reason not in gonderilen_araclar_db[track_id]:
                        kuyruaga_ekle = True
                    
                    # 2. Gönderilmiş ama plakası yeni okunduysa -> Kuyruğa Ekle (Update için)
                    elif plaka_okundu and (durum_anahtari not in gonderilen_plakali_araclar):
                        kuyruaga_ekle = True

                    if kuyruaga_ekle:
                        # Veritabanına bu sebebi daha sonra tekrar eklememesi için işaretle
                        # (NOT: Gerçek gönderim 3 sn sonra olacak ama "planlandı" olarak işaretliyoruz)
                        gonderilen_araclar_db[track_id].add(arac_reason)
                        if plaka_okundu:
                            gonderilen_plakali_araclar.add(durum_anahtari)

                        # --- BURASI KRİTİK NOKTA ---
                        # Eğer araç zaten bekleme listesindeyse, sadece plaka bilgisini güncelle.
                        # Eğer bekleme listesinde değilse, "kare.copy()" ile FOTOĞRAFI DONDURUP ekle.
                        
                        veriler = {
                                'plate_text': final_plaka, 
                                'reason': arac_reason,
                                'camera_name': MEVCUT_KAMERA_ADI,
                                'track_id': str(track_id),
                                'status': arac_status
                            }

                        if track_id not in bekleyen_yuklemeler:
                            # İLK GÖRÜLME ANI: Fotoğrafı çek (kopyala) ve zamanı başlat
                            print(f"⏳ ID:{track_id} kuyruğa alındı. {VERITABANI_GONDERIM_GECIKMESI}s bekleniyor...")
                            bekleyen_yuklemeler[track_id] = {
                                'image': kare.copy(),  # <--- DOĞRU FOTOĞRAF BURADA SAKLANIYOR
                                'data': veriler,
                                'first_seen': time.time(),
                                'last_reason': durum_anahtari # Tekrar kontrol için
                            }
                        else:
                            # ZATEN BEKLİYOR: Sadece plaka verisi güzelleşirse güncelle
                            # Fotoğrafı güncelleme! Çünkü ilk ihlal anı daha önemli olabilir.
                            if plaka_okundu and bekleyen_yuklemeler[track_id]['data']['plate_text'] == "OKUNAMADI":
                                print(f"♻️ ID:{track_id} beklerken plakası netleşti: {final_plaka}")
                                bekleyen_yuklemeler[track_id]['data']['plate_text'] = final_plaka
                                # Plaka eklendiği için plakalılar listesine de ekleyelim
                                gonderilen_plakali_araclar.add(durum_anahtari)


        # --- KUYRUKTAKİ BEKLEYENLERİ KONTROL ET VE GÖNDER ---
        # --- KUYRUKTAKİ BEKLEYENLERİ KONTROL ET VE GÖNDER ---
        for tid, info in list(bekleyen_yuklemeler.items()):
            gecen_sure = simdiki_zaman - info['first_seen']
            
            if gecen_sure >= VERITABANI_GONDERIM_GECIKMESI:
                print(f"🚀 ID:{tid} için bekleme süresi doldu. Sunucuya gönderiliyor...")

                # ==============================================================================
                # DÜZELTME BURADA: Göndermeden hemen önce EN GÜNCEL oylama sonucunu al!
                # ==============================================================================
                guncel_en_iyi_plaka = arac_plaka_son_karar.get(tid)

                # Eğer sistem 5 saniye boyunca daha iyi bir okuma yaptıysa, veriyi güncelle
                if guncel_en_iyi_plaka and guncel_en_iyi_plaka != "OKUNAMADI":
                    print(f"   ♻️ Veri güncelleniyor: {info['data']['plate_text']} -> {guncel_en_iyi_plaka}")
                    info['data']['plate_text'] = guncel_en_iyi_plaka
                # ==============================================================================
                
                # Resmi hazırla
                basarili_kodlama, tampon = cv2.imencode(".jpg", info['image']) # Saklanan (ilk ihlal anı) resmi kullan
                if basarili_kodlama:
                    resim_byte_dizisi = io.BytesIO(tampon)
                    resim_byte_dizisi.seek(0)
                    dosyalar = {'image': ('violation.jpg', resim_byte_dizisi, 'image/jpeg')}
                    
                    try:
                        print(f"   📤 Gönderilen Veri -> Plaka: {info['data']['plate_text']} | Sebep: {info['data']['reason']}")
                        response = requests.post(SUNUCU_API_ADRESI, data=info['data'], files=dosyalar, timeout=5)
                        
                        if response.status_code in [200, 201]:
                            print(f"   ✅ Sunucu Yanıtı: {response.json().get('message', 'Başarılı')}")
                        else:
                            print(f"   ❌ Sunucu Hatası: {response.status_code}")
                    
                    except Exception as e:
                        print(f"   ❌ Bağlantı Hatası: {e}")
                
                # Listeden sil (Görev tamamlandı)
                del bekleyen_yuklemeler[tid]


        # Çıkış Kontrolü
        cikan_araclar = onceki_karedeki_idler - mevcut_karedeki_idler
        for cikan_id in cikan_araclar:
            if cikan_id in arac_plaka_havuzu:
                sonuc_plaka = Counter(arac_plaka_havuzu[cikan_id]).most_common(1)[0][0]
                print(f"--- 🏁 ARAÇ ÇIKIŞ YAPTI (ID: {cikan_id}) -> PLAKA: {sonuc_plaka}")
            else:
                print(f"--- 🏁 ARAÇ ÇIKIŞ YAPTI (ID: {cikan_id}) -> PLAKA: OKUNAMADI")
            print("-------------------------------------------\n")

        onceki_karedeki_idler = mevcut_karedeki_idler.copy()
        son_islenen_kare = kare.copy()

    # GOSTERIM
    gosterilecek_kare = son_islenen_kare if son_islenen_kare is not None else kare
    gosterilecek_kare = cv2.resize(gosterilecek_kare, (1280, 720))
    cv2.imshow(f"Yaya Yolu (EasyOCR) - {MEVCUT_KAMERA_ADI}", gosterilecek_kare)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_kaynagi.release()
cv2.destroyAllWindows()