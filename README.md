# FER Hotel Smartcam (HOG + SVM)

Skripsi — pengenalan emosi wajah real-time (`happy`/`sad`/`angry`/`neutral`),
edge computing, semua lokal. Detail rancangan: [CLAUDE.md](CLAUDE.md).

## Status
Tahap 1-8 selesai (scaffold, capture, deteksi wajah, HOG, dataset, training,
evaluasi, classifier, logger, UI Streamlit). Sisa: uji end-to-end di Docker.

## Dataset
AffectNet format YOLO (`fatihkgg/affectnet-yolo-format`), 500 citra per kelas
untuk 4 kelas dalam scope: happy, sad, angry, neutral.

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('fatihkgg/affectnet-yolo-format'))"
python -c "
from pathlib import Path
from src import dataset_manager as dm, db
P = Path('<path hasil unduhan>')/'YOLO_format/train'
dm.impor_yolo(P, cap=500)          # salin ke data/raw/<label>/
dm.ingest(dm.RAW, 'publik', db.connect())
"
```

## Jalan lokal (dev di Mac — webcam tidak bisa lewat Docker)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Cek modul (tiap file punya self-check di `__main__`):
```bash
python src/db.py
python -m src.face_detection
```
Uji kamera + deteksi wajah, jendela OpenCV (izinkan akses kamera saat diminta macOS):
```bash
python demo_kamera.py
```
Jalankan antarmuka:
```bash
streamlit run app/main.py
```

## Jalan di Docker (Linux / RasPi)
Uncomment blok `devices:` di `docker/docker-compose.yml` dulu, lalu:
```bash
docker compose -f docker/docker-compose.yml up --build
```
Buka http://localhost:8501

`data/` dan `models/` di-mount sebagai volume — data persist walau container dihapus.
