"""Unduh dataset AffectNet lalu masukkan ke sistem — hanya perlu dijalankan bila
ingin melatih ulang model. Untuk memakai sistem, langkah ini tidak dibutuhkan
karena model terlatih sudah disertakan.

Jalankan: python siapkan_dataset.py
"""
from pathlib import Path

from src import dataset_manager as dm, db

DATASET = "fatihkgg/affectnet-yolo-format"
CAP = 500          # citra per kelas; seimbang antar kelas dan waktu training wajar


def unduh():
    """Ambil dataset dari Kaggle. Tidak perlu akun. Sekitar 258 MB."""
    import kagglehub
    print(f"Mengunduh {DATASET} (sekitar 258 MB, hanya sekali)...")
    return Path(kagglehub.dataset_download(DATASET)) / "YOLO_format" / "train"


def main():
    sumber = unduh()
    print(f"Menyalin {CAP} citra per kelas ke data/raw/ ...")
    print("  ", dm.impor_yolo(sumber, cap=CAP))

    print("Praproses (deteksi wajah -> grayscale 128x128) dan pencatatan ke database...")
    conn = db.connect()
    baru = dm.ingest(dm.RAW, "publik", conn)
    print("   citra baru tercatat:", baru)

    print("\nIsi dataset sekarang:")
    for r in dm.ringkasan(conn):
        print(f"   {r['label_emosi']:<9} {r['sumber_data']:<9} {r['jumlah']} citra")
    print("\nSelesai. Model bisa dilatih ulang dari halaman Admin, "
          "atau tetap gunakan model yang sudah ada.")


if __name__ == "__main__":
    main()
