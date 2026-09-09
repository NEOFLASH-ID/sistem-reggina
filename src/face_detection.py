"""UC2: Deteksi Wajah — Haar Cascade, satu wajah saja (scope skripsi)."""
import cv2
import numpy as np

from .preprocessing import SIZE, praproses
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect(frame):
    """Kembalikan bbox (x, y, w, h) wajah terbesar, atau None kalau tak ada."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # terbesar = subjek di depan kamera


def crop(frame, box):
    """Potong area wajah lalu praproses -> grayscale 128x128, siap untuk HOG."""
    x, y, w, h = box
    return praproses(frame[max(y, 0):y + h, max(x, 0):x + w])


def detect_and_crop(frame):
    """Jalur pakai sehari-hari: frame masuk, citra 128x128 keluar (atau None)."""
    box = detect(frame)
    return None if box is None else crop(frame, box)


if __name__ == "__main__":
    assert not _cascade.empty(), "file haarcascade tidak ketemu"

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    assert detect(blank) is None, "citra kosong seharusnya tanpa wajah"
    assert detect_and_crop(blank) is None

    out = crop(blank, (10, 10, 50, 80))
    assert out.shape == (SIZE, SIZE), out.shape
    assert out.ndim == 2, "hasil crop harus grayscale"
    print("face_detection OK")
