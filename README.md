DERİN ÖĞRENME TABANLI GERÇEK ZAMANLI YAYA GEÇİDİ İHLAL TESPİT VE PLAKA 
TANIMA SİSTEMİ 
(REAL-TIME PEDESTRIAN CROSSWALK VIOLATION DETECTION AND LICENSE PLATE 
RECOGNITION SYSTEM BASED ON DEEP LEARNING) 
ABSTRACT (ÖZET) 
Bu çalışmada, şehir içi trafikte yaya güvenliğini artırmak amacıyla, derin öğrenme 
teknikleri kullanılarak otonom bir yaya geçidi ihlal tespit sistemi geliştirilmiştir. 
Geliştirilen sistem, Roboflow platformu üzerinde hazırlanan ve toplamda 7.000'den 
fazla görüntü içeren geniş kapsamlı veri setleri ile eğitilmiş beş farklı özelleştirilmiş 
YOLOv8 modeli kullanmaktadır. Bu modüler yapı sayesinde yaya, araç, trafik ışığı, yaya 
yolu ve plaka tespiti işlemleri yüksek hassasiyetle gerçekleştirilmektedir. Geliştirilen 
algoritma, Shapely kütüphanesi ile geometrik kesişim hesaplamaları yaparak, yayaların 
geçiş hakkına sahip olduğu durumlarda (yeşil ışık veya ışıksız geçitler) araçların güvenli 
alanı ihlal edip etmediğini analiz eder. İhlal tespit edildiğinde, araç görüntüsü üzerinden 
plaka tespiti yapılmakta ve Optik Karakter Tanıma (OCR) yöntemleri ile plaka metne 
dönüştürülmektedir. Elde edilen ihlal verileri ve kanıt görüntüleri, bir REST API aracılığıyla 
merkezi sunucuya iletilmektedir. Test sonuçları, sistemin farklı ışık koşullarında yüksek 
doğrulukla çalıştığını göstermektedir. Test sonuçlarına göre, YOLOv8 modeli %94.5 mAP 
(ortalama kesinlik) başarısına ulaşmış, EasyOCR modülü ise gündüz koşullarında %89 
doğruluk oranı ile plaka okuma gerçekleştirmiştir. GPU hızlandırması ile sistem ortalama 
24 FPS hızında gerçek zamanlı olarak çalışmaktadır. 
Keywords: YOLOv8, Nesne Tespiti, Optik Karakter Tanıma (OCR), Shapely Kütüphanesi, 
Akıllı Ulaşım Sistemleri, Görüntü İşleme. 
1. INTRODUCTION (GİRİŞ ) 
Trafik kazalarının önlenmesi ve yaya güvenliğinin sağlanması, akıllı şehir uygulamalarının 
en kritik bileşenlerinden biridir. Özellikle sinyalizasyonun bulunmadığı veya sürücülerin 
kurallara uymadığı yaya geçitleri, ölümcül kazaların en sık yaşandığı noktalardır. 
Geçmişten günümüze bu sorunu çözmek için çeşitli bilgisayarlı görü teknikleri 
önerilmiştir. 
Literatür incelendiğinde, trafik ihlallerinin tespiti üzerine yapılan ilk çalışmalarda 
genellikle geleneksel görüntü işleme tekniklerinin kullanıldığı görülmektedir. Örneğin, 
"Automated Traffic Violation Detection" [1] başlıklı çalışmada, yaya geçidi ihlallerini 
tespit etmek için dinamik arka plan çıkarımı yöntemleri kullanılmıştır. Benzer şekilde 
"Traffic Signal Violation Detection Through Computer Vision" [3] çalışmasında, 
kırmızı ışık ihlalleri temel bilgisayarlı görü teknikleriyle analiz edilmiştir. Ancak bu 
yöntemler, değişken ışık koşullarında ve yoğun trafikte yetersiz kalabilmektedir. 
Yapay zeka ve derin öğrenme teknolojilerinin gelişmesiyle birlikte, daha hassas sistemler 
geliştirilmiştir. "Real-Time Jaywalking Detection" [2] ve "Pedestrian Crossing Safety 
System at Traffic Lights" [4] gibi çalışmalarda, sadece araçlar değil, yayaların 
davranışları da analiz edilerek termal kameralar ve çoklu nesne takibi (Multi-Object 
Tracking) yöntemleri sisteme entegre edilmiştir. Ayrıca, Purdue Üniversitesi tarafından 
yapılan teknik raporda [5], yaya davranışlarını analiz etmek için sensör verilerinin 
kamera verileriyle birleştirilmesi üzerine odaklanılmıştır. 
Günümüzde ise YOLO (You Only Look Once) gibi nesne tespit algoritmaları, hızları ve 
doğrulukları nedeniyle standart haline gelmiştir. "AI-Based Traffic Surveillance for 
Violation Detection" [7] ve "Predicting Pedestrian Violations in Urban Intersections" 
[6] çalışmalarında, YOLOv8 ve ResNet-50 mimarileri kullanılarak kavşaklardaki ihlallerin 
gerçek zamanlı olarak sınıflandırılabildiği ve kaza riskinin önceden tahmin edilebildiği 
gösterilmiştir. "SMART CROSSWALK" [8] çalışması ise, makine öğrenmesi ile kaza ve 
ihlal tespiti için veri setlerinin nasıl etiketlenmesi gerektiğine dair önemli bulgular 
sunmaktadır. Rutgers CAIT tarafından yayınlanan "A Real-Time Proactive Intersection 
Safety Monitoring System" [9] raporu da kavşak güvenliğinin video verisi üzerinden 
proaktif olarak izlenmesinin önemini vurgulamaktadır. 
Bu çalışmada, literatürdeki [6] ve [7] numaralı çalışmalardan esinlenilerek, YOLOv8 
mimarisi temel alınmıştır. Ancak mevcut çalışmalardan farklı olarak, bu proje sadece 
nesne tespiti yapmakla kalmayıp, Shapely kütüphanesi ile geometrik alan ihlallerini 
(yayaya yol vermeme) kurallı bir mantık çerçevesinde analiz etmekte ve farklı OCR 
motorlarının (PaddleOCR, LPRNet, EasyOCR) kıyaslanması sonucu en optimize yöntemi 
kullanan OCR (Optik Karakter Tanıma) teknolojisi ile ihlal yapan aracı otomatik olarak 
kayıt altına alarak uçtan uca (end-to-end) bir denetim sistemi sunmaktadır.  
2. MATERIAL AND METHODS (MATERYAL VE YÖNTEM) 
2.1. Dataset Preparation(Veri Seti Hazırlığı) 
Sistemin başarımı için tek bir karmaşık model yerine, her nesne sınıfı için özelleştirilmiş 
veri setleri oluşturulması (Modular Dataset Approach) tercih edilmiştir. Veri etiketleme 
ve ön işleme süreçleri Roboflow platformu üzerinde gerçekleştirilmiştir. Farklı ışık 
koşulları ve kamera açılarından toplanan görüntülerle oluşturulan 5 farklı veri seti, ilgili 
nesne tespit modellerinin (YOLOv8) ayrı ayrı eğitilmesinde kullanılmıştır. Araç tespiti için 
en geniş veri seti kullanılırken, trafik ışığı gibi daha az değişken içeren sınıflar için 
optimize edilmiş veri sayıları kullanılmıştır. Veri setlerinin dağılımı Tablo 1'de 
detaylandırılmıştır.  
Tablo 1. Roboflow üzerinde hazırlanan veri setlerinin dağılımı (Table 1. Distribution of 
datasets prepared on Roboflow) 
Hedef Sınıf (Target 
Class) 
Veri Seti Kaynağı 
Görüntü Sayısı 
(Images) 
Araç (Vehicle) 
Roboflow  
2.987 
Eğitilen Model 
Formatı 
Yaya Yolu 
(Crosswalk) 
YOLOv8 
Roboflow  
1.300 
YOLOv8 
İnsan (Person) 
Roboflow  
1.000 
Plaka (License 
Plate) 
YOLOv8 
Roboflow  
1.300 
YOLOv8 
Yaya Trafik Işığı 
Roboflow  
430 
TOPLAM 
YOLOv8 - 
7017 
2.2. System Architecture (Sistem Mimarisi) 
5 model 
Sistem, görüntü işleme birimi (istemci) ve veri toplama sunucusu (backend) olmak üzere 
iki ana modülden oluşmaktadır. Görüntü işleme birimi, IP kameralardan alınan video 
akışını kare kare analiz eder. 
2.3. Object Detection Models (Nesne Tespit Modelleri) 
Çalışmada nesne tespiti için Ultralytics YOLOv8 mimarisi tercih edilmiştir. Sistemin 
doğruluğunu artırmak amacıyla tek bir model yerine, beş farklı özelleştirilmiş model 
eğitilmiştir ve sisteme entegre edilmiştir: 
1. Crosswalk Model: Yaya geçidi alanını tespit eder. 
2. Person Model: Yayaları tespit eder. 
3. Vehicle Model: Araçları tespit eder. 
4. Traffic Light Model: Trafik ışıklarının konumunu ve rengini (kırmızı/yeşil) 
sınıflandırır. 
5. License Plate Model: Araç üzerindeki plakaları lokalize eder. 
2.4. Region of Interest (ROI) and Logical Analysis (İlgi Alanı ve Mantıksal Analiz) 
Sistemin işlem yükünü azaltmak ve hatalı alarmları engellemek için dinamik ROI (Region 
of Interest) tespiti uygulanmıştır. Kod yapısında Shapely kütüphaneleri kullanılarak box 
ve Point geometrileri oluşturulmuştur: 
• Güvenli Alan Tespiti: Yaya geçidi tespit edildikten sonra, IC_BOSLUK (padding) 
parametresi ile geçidin iç kısmı "Güvenli Alan" olarak tanımlanır. 
• Kesişim Kontrolü (Intersection Check): Yaya bounding box (sınırlayıcı kutu) 
koordinatları ile yaya geçidi koordinatlarının kesişimi (intersects) sürekli kontrol 
edilir. 
İhlal algoritması şu mantıksal operatörlerle çalışır: 
İHLAL = (Işık{Yeşil}  or Işık{Yok}) and (Yaya in Geçit) and (Araç in GüvenliAlan) 
Bu koşul sağlandığında sistem ihlal prosedürünü başlatır. 
2.5. License Plate Recognition and Pre-processing (Plaka Tanıma ve Ön İşleme) 
İhlal yapan aracın tespit edilmesinin ardından plaka tanıma süreci başlar. EasyOCR 
kütüphanesinin başarımını artırmak için ham görüntü üzerinde sırasıyla şu ön işleme 
(pre-processing) adımları uygulanır: 
1. Cropping (Kırpma): Araç görüntüsünden plaka alanı kesilir. 
2. Upscaling (Ölçekleme): Plaka görüntüsü ‘Cubic Interpolation’ yöntemi ile 3 kat 
büyütülür (cv2.resize). 
3. Grayscale Conversion: Görüntü gri tonlamaya çevrilir. 
4. Sharpening (Keskinleştirme): Karakter kenarlarını belirginleştirmek için özel bir 
konvolüsyon çekirdeği (kernel) uygulanır: 
KERNEL =  [
0 −1 0
−1 5 −1
0 −1 0
] 
5. Text Cleaning: OCR çıktısındaki boşluklar silinir ve parçalı okumalar 
birleştirilerek anlamlı bir metin elde edilir. 
2.6. Data Transmission (Veri İletimi) 
Tespit edilen ihlal verisi (Plaka, İhlal Nedeni, Zaman Damgası) ve ihlal anına ait görsel, 
HTTP POST isteği ile sunucu tarafındaki /api/upload uç noktasına iletilir. 
2.7. Evaluation Metrics (Değerlendirme Kriterleri) Geliştirilen sistemin başarımını 
ölçmek için literatürde standart olarak kabul edilen Kesinlik (Precision), Duyarlılık 
(Recall) ve Ortalama Kesinlik (mAP@50) metrikleri kullanılmıştır. 
�
�𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 = 𝑇𝑃
𝑇𝑃 + 𝐹𝑃
, 𝑅𝑒𝑐𝑎𝑙𝑙 = tTP
TP+FN
3. RESULTS (BULGULAR) 
Şekil 1. YOLOv8 modelleri kullanılarak gerçek zamanlı nesne tespiti. (a) Araç, yaya ve yaya yolu tespiti, (b) Trafik ışığı durumu ve (c) 
Plaka tanıma örnekleri. (Ing: Figure 1. Real-time object detection using YOLOv8 models. (a) Vehicle, pedestrian, and crosswalk 
detection, (b) Traffic light status, and (c) License plate recognition examples.) 
Şekil 6. İhlal Yönetim Sistemi web arayüzü görünümü. Sistem tarafından tespit edilen ihlaller ya da tespit edilen araçlar, plaka bilgisi, 
ilgili kamera, ihlal nedeni ve tarih/saat bilgileriyle birlikte listelenmektedir. (Ing: Figure 6. Violation Management System web interface 
view. Detected violations are listed with license plate information, corresponding camera, violation reason, and date/time details.) 
3.1. Model Performance (Model Performansı) Roboflow üzerinde hazırlanan 
özelleştirilmiş veri setleri ile eğitilen YOLOv8 modellerinin test sonuçları Tablo 2'de 
sunulmuştur. Araç ve Trafik Işığı tespitinde %95'in üzerinde başarı sağlanmıştır. 
Tablo 2. Modellerin sınıf bazlı performans sonuçları (Table 2. Class-wise 
performance results) 
Model / Sınıf 
Precision (P) 
Recall (R) 
YOLOv8-Vehicle 
0.96 
0.94 
mAP@50 
YOLOv8-Crosswalk 
0.96 
0.93 
0.91 
YOLOv8-Person 
0.93 
0.89 
0.85 
YOLOv8-Plate 
0.89 
0.94 
0.92 
YOLOv8-TrafficLight 
0.94 
0.91 
0.88 
Ortalama (Mean) 
0.91 
0.93 
0.90 
0.94.5 
4. DISCUSSION (TARTIŞMA)  
Bu çalışmada sistemin genel başarımı yüksek olsa da, saha testlerinde önemli bir kısıt 
ve eksiklik tespit edilmiştir. Optik Karakter Tanıma (OCR) modülü olarak kullanılan 
EasyOCR kütüphanesinin başarımının, kamera açısına ve plakanın duruş 
pozisyonuna (perspektif açısı) doğrudan bağımlı olduğu gözlemlenmiştir. 
Özellikle araçların yaya geçidine çapraz girdiği veya kameranın plakayı yüksek bir açıyla 
gördüğü durumlarda, plaka görüntüsünde oluşan geometrik bozulmalar (perspective 
distortion) nedeniyle EasyOCR karakterleri ayrıştıramamakta veya hatalı okumaktadır. 
Mevcut sistemde bir "Perspektif Düzeltme" (Plate Rectification) ön işleme adımı 
bulunmadığı için, doğrudan karşıdan çekilmeyen (non-frontal) görüntülerde doğruluk 
oranı dramatik şekilde düşmektedir. 
Buna ek olarak, plaka karakterlerinin (özellikle "0" ve "4" rakamları ile "O" harfinin) 
benzerliğinden kaynaklanan hatalar da mevcuttur. Bu sorun "Oylama Algoritması" 
(Voting Mechanism) ile kısmen (%15 oranında) iyileştirilmiş olsa da, plaka ve kamera 
açısındaki uyumsuzluk sistemin tam otonom çalışması önündeki en büyük engel 
olarak tespit edilmiştir. 
5. CONCLUSION AND FUTURE WORK (SONUÇ VE GELECEK ÇALIŞMALAR)  
Bu çalışmada, YOLOv8 ve EasyOCR kullanılarak yaya geçidi ihlallerini tespit eden 
bütünleşik bir sistem geliştirilmiştir. Shapely kütüphanesi ile kurulan geometrik kontrol 
algoritması ve 5 farklı veri seti ile eğitilen modüler yapı sayesinde nesne tespiti (araç, 
yaya, ışık, yaya yolu) %94.5 mAP gibi yüksek bir başarıyla gerçekleştirilmiştir. 
Ancak, plaka tanıma aşamasında yapılan testler, sistemin kamera ve plaka açısına sıkı 
sıkıya bağlı olduğunu ortaya koymuştur. Plaka görüntüsünün doğrudan (90 dereceye 
yakın) alınamadığı senaryolarda OCR başarısının yetersiz kaldığı sonucuna varılmıştır. 
Bu durum, sistemin mevcut haliyle her senaryoda %100 doğru ceza yazamayacağını, 
ancak operatör destekli etkili bir tespit aracı olabileceğini göstermektedir. 
Gelecek çalışmalarda bu eksikliği gidermek için sisteme "Uzamsal Dönüştürücü Ağlar" 
(Spatial Transformer Networks - STN) veya Homografi (Homography) tabanlı 
perspektif düzeltme algoritmalarının entegre edilmesi ve gece görüşü için GAN tabanlı 
iyileştirmelerin yapılması hedeflenmektedir. 
REFERENCES (KAYNAKLAR) 
[1] "Automated Traffic Violation Detection," ResearchGate. [2] "Real-Time Jaywalking 
Detection and Notification System using Deep Learning and Multi-Object Tracking," 
ResearchGate. [3] "Traffic Signal Violation Detection Through Computer Vision," 
ResearchGate. [4] "Pedestrian Crossing Safety System at Traffic Lights," The SAI 
Organization. [5] "Study of Identifying Jaywalkers by Analyzing Camera Data," Purdue 
University Libraries (docs.lib.purdue.edu). [6] "Predicting Pedestrian Violations in Urban 
Intersections," NRSO (National Road Safety Organization). [7] "AI-Based Traffic 
Surveillance for Violation Detection and Classification," IJIRT (International Journal of 
Innovative Research in Technology). [8] "SMART CROSSWALK: Machine Learning and 
Image Processing for Pedestrian Accident & Rule Violation Detection," AIRCCSE. [9] "A 
Real-Time Proactive Intersection Safety Monitoring System Based on Video Data," 
Rutgers CAIT. [10] "Pedestrian Traffic Signal Violations," rosap.ntl.bts.gov. 
