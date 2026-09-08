const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
// Dosya adının 'violation' (küçük harf) olduğundan emin olun
const Violation = require('../models/violation');

const uploadDir = path.join(__dirname, '../public/uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

// --- VERİ KAYDETME VE GÜNCELLEME ---
router.post('/upload', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'Resim yok.' });

        const { track_id, plate_text, reason, camera_name, status } = req.body;

        // 1. KONTROL: Bu araç (track_id) son 60 saniyede geçti mi?
        const existingRecord = await Violation.findOne({
            trackId: track_id,
            timestamp: { $gte: new Date(Date.now() - 60000) } 
        });

        // EĞER KAYIT VARSA:
        if (existingRecord) {
            
            // SENARYO A: Eski kayıt "OKUNAMADI" ama şimdi gerçek plaka geldi -> GÜNCELLE
            if (existingRecord.plateText === "OKUNAMADI" && plate_text !== "OKUNAMADI") {
                
                // Eski (plakasız) resmi sil (Disk tasarrufu)
                if (existingRecord.imageFilename) {
                    fs.unlink(path.join(uploadDir, existingRecord.imageFilename), (err) => {});
                }

                // Kaydı yeni bilgilerle güncelle
                existingRecord.plateText = plate_text;
                existingRecord.imageFilename = req.file.filename; // Yeni resmi koy
                await existingRecord.save();

                console.log(`♻️ [GÜNCELLEME] ID:${track_id} plakası "${plate_text}" olarak düzeltildi.`);
                return res.status(200).json({ success: true, message: 'Updated' });
            }

            // SENARYO B: Zaten plakası var veya yeni gelen de okunamadı -> REDDET
            // Yeni yüklenen gereksiz resmi sil
            fs.unlink(req.file.path, (err) => {});
            
            console.log(`⚠️ [SPAM KORUMA] ID:${track_id} zaten kayıtlı. İşlem atlandı.`);
            // Python'a 200 dönüyoruz ki hata sanıp tekrar tekrar denemesin
            return res.status(200).json({ success: true, message: 'Already exists' });
        }

        // 2. YENİ KAYIT OLUŞTURMA (Eğer geçmişte kaydı yoksa)
        const newEntry = new Violation({
            plateText: plate_text || "OKUNAMADI",
            imageFilename: req.file.filename,
            reason: reason,
            cameraName: camera_name,
            trackId: track_id,
            status: status
        });

        await newEntry.save();
        console.log(`✅ [${(status || 'BELIRSIZ').toUpperCase()}] ID:${track_id} Plaka:${plate_text} kaydedildi.`);
        res.status(201).json({ success: true });

    } catch (err) {
        console.error("❌ Kayıt Hatası:", err);
        res.status(500).json({ error: err.message });
    }
});

// --- VERİ LİSTELEME ---
router.get('/violations', async (req, res) => {
    try {
        const { camera, search } = req.query;
        let query = {};

        if (camera && camera !== 'all') query.cameraName = camera;
        if (search) query.plateText = { $regex: search, $options: 'i' };

        const totalCount = await Violation.countDocuments(query);
        const violationCount = await Violation.countDocuments({...query, status: 'violation'});
        const records = await Violation.find(query).sort({ timestamp: -1 }).limit(100);

        res.json({ records, totalCount, violationCount }); 
    } catch (err) {
        res.status(500).json({ error: 'Veri çekilemedi' });
    }
});

module.exports = router;