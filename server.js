const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const Violation = require('./models/violation');
const app = express();
const PORT = 2000;

// --- AYARLAR ---
app.use(cors());
app.use(express.json()); // JSON verilerini okumak için
app.use(express.urlencoded({ extended: true })); // Form verilerini okumak için

// Statik Dosyalar (CSS, JS, Resimler)
// 'public' klasörünü dışarıya açıyoruz
app.use(express.static('public'));

// View Engine (EJS)
app.set('view engine', 'ejs');

// --- VERİTABANI BAĞLANTISI ---
// Yerel MongoDB sunucusuna bağlanır
mongoose.connect('mongodb://127.0.0.1:27017/trafficDB')
    .then(() => console.log('✅ MongoDB Bağlantısı Başarılı'))
    .catch(err => console.error('❌ MongoDB Hatası:', err));

// --- ROTALAR ---
// api.js dosyasını /api altına bağlıyoruz
// Python buraya istek atacak: http://127.0.0.1:2000/api/upload
app.use('/api', require('./routes/api'));

app.get('/', async (req, res) => {
    try {
        // 1. URL'den seçilen kamerayı al
        const selectedCamera = req.query.camera_name;

        // 2. Veritabanı sorgusunu hazırla
        let dbQuery = {};
        
        // EKRAN GÖRÜNTÜNE GÖRE DÜZELTME BURADA: 'cameraName' kullanıldı.
        if (selectedCamera && selectedCamera !== 'all') {
            dbQuery.cameraName = selectedCamera; 
        }

        // 3. İhlal verilerini çek (Tabloyu doldurmak için)
        // Veritabanında verilerin olduğunu görüyorum, bunları çekiyoruz.
        const violations = await Violation.find(dbQuery).sort({ _id: -1 });

        // 4. Kamera listesini çek (Dropdown için)
        // EKRAN GÖRÜNTÜNE GÖRE DÜZELTME: 'camera_name' yerine 'cameraName'
        const cameras = await Violation.distinct('cameraName');

        console.log("Bulunan Kameralar:", cameras); // Konsola yazdırıp kontrol et

        // 5. Sayfayı render et
        res.render('index', { 
            data: violations,       
            cameras: cameras,       
            currentFilter: selectedCamera 
        });

    } catch (err) {
        console.error("Ana sayfa yükleme hatası:", err);
        res.render('index', { data: [], cameras: [], currentFilter: null });
    }
});

// Sunucuyu Başlat
app.listen(PORT, () => {
    console.log(`🚀 Sunucu çalışıyor: http://localhost:${PORT}`);
});