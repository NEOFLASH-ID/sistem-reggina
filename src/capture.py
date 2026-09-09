"""UC1: Menangkap Citra Wajah — buka kamera, hasilkan frame satu per satu."""
import cv2


def frames(source=0):
    """Generator frame BGR dari webcam. Selalu melepas kamera saat selesai."""
    cam = cv2.VideoCapture(source)
    ok, frame = cam.read() if cam.isOpened() else (False, None)
    if not ok:
        cam.release()
        raise RuntimeError(
            f"Kamera {source} tidak menghasilkan frame. Tiga penyebab tersering di macOS:\n"
            "1. Izin kamera belum diberikan: System Settings > Privacy & Security > "
            "Camera, aktifkan untuk app ini, lalu quit app (Cmd+Q) dan buka lagi.\n"
            "2. Program dijalankan dari app lain yang tidak punya izin kamera — izin "
            "macOS melekat pada aplikasi induk, bukan pada skrip Python-nya. "
            "Jalankan dari terminal yang sudah diizinkan.\n"
            "3. Kamera sedang dipakai proses lain (Zoom, Photo Booth, skrip lain); "
            "macOS hanya memberikannya ke satu proses.")
    # ponytail: webcam macOS/USB sesekali drop frame; sabar 30 kali gagal beruntun
    # (~1 detik) sebelum menyerah. Naikkan kalau kamera di lokasi penelitian lebih rewel.
    try:
        gagal = 0
        while gagal < 30:
            if ok:
                gagal = 0
                yield frame
            else:
                gagal += 1
            ok, frame = cam.read()
    finally:
        cam.release()


def snapshot(source=0):
    """Ambil satu frame saja (dipakai halaman admin / pengujian)."""
    return next(frames(source), None)


if __name__ == "__main__":
    for i, f in enumerate(frames()):
        cv2.imshow("capture (q = keluar)", f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
