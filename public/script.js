document.addEventListener('DOMContentLoaded', () => {
    const cameraSelect = document.getElementById('camera-select');
    const searchInput = document.getElementById('search-input');
    
    fetchData(); // Başlat
    
    cameraSelect.addEventListener('change', fetchData);
    searchInput.addEventListener('input', fetchData);
    
    // Her 3 saniyede bir yenile
    setInterval(fetchData, 3000);
});

async function fetchData() {
    const camera = document.getElementById('camera-select').value;
    const search = document.getElementById('search-input').value;

    try {
        const response = await fetch(`/api/violations?camera=${camera}&search=${search}`);
        const data = await response.json();

        // İstatistikleri Güncelle (HTML'de bu ID'leri ekleyeceğiz)
        document.getElementById('total-pass-count').innerText = data.totalCount;
        document.getElementById('violation-count').innerText = data.violationCount;

        renderGallery(data.records);
    } catch (error) {
        console.error('Hata:', error);
    }
}

document.getElementById('camera-select').addEventListener('change', function() {
        const val = this.value;
        const url = new URL(window.location.href);
        if (val === 'all') url.searchParams.delete('camera_name');
        else url.searchParams.set('camera_name', val);
        window.location.href = url.toString();
    });

function renderGallery(records) {
    const gallery = document.getElementById('main-gallery');
    gallery.innerHTML = ''; // Temizle

    records.forEach(item => {
        const card = document.createElement('div');
        // Eğer ihlalse 'card-violation', normalse 'card-normal' sınıfı ekle
        const typeClass = item.status === 'violation' ? 'violation-border' : 'normal-border';
        
        card.className = `card ${typeClass}`;
        
        const date = new Date(item.timestamp).toLocaleTimeString('tr-TR');
        const plaka = item.plateText === 'OKUNAMADI' ? '---' : item.plateText;
        const durumYazisi = item.status === 'violation' ? '⚠️ İHLAL' : '✅ NORMAL';
        const durumRengi = item.status === 'violation' ? 'red' : 'green';

        card.innerHTML = `
            <div class="card-image">
                <img src="/uploads/${item.imageFilename}" loading="lazy">
                <span class="plate-tag" style="background:${item.status === 'violation' ? '#c0392b' : '#27ae60'}">
                    ${plaka}
                </span>
            </div>
            <div class="card-info">
                <h3 style="color:${durumRengi}; margin-bottom:5px;">${durumYazisi}</h3>
                <p><strong>ID:</strong> #${item.trackId || '?'}</p>
                <p><strong>Neden:</strong> ${item.reason}</p>
                <p class="timestamp">🕒 ${date}</p>
            </div>
        `;
        gallery.appendChild(card);
    });
}