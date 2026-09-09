"""UC4: Klasifikasi Emosi — muat model SVM tersimpan, prediksi dari citra wajah."""
import joblib
import numpy as np

from .face_detection import detect_and_crop
from .feature_extraction import extract
from .training import MODEL_PATH

_cache = {}


def muat(path=MODEL_PATH):
    """Muat model sekali lalu simpan di memori — real-time tidak boleh baca disk tiap frame."""
    path = str(path)
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]


def prediksi(citra_wajah, model=None):
    """Citra wajah 128x128 -> (label, confidence 0..1)."""
    model = model or muat()
    fitur = extract(citra_wajah).reshape(1, -1)
    peluang = model.predict_proba(fitur)[0]
    i = int(np.argmax(peluang))
    return model.classes_[i], float(peluang[i])


def prediksi_frame(frame, model=None):
    """Frame kamera utuh -> (label, confidence, bbox) atau None kalau tak ada wajah."""
    from .face_detection import detect
    box = detect(frame)
    if box is None:
        return None
    from .face_detection import crop
    label, conf = prediksi(crop(frame, box), model)
    return label, conf, box


if __name__ == "__main__":
    import cv2
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 0.2, (10, 8100)) for m in (0, 5)])
    palsu = SVC(probability=True, random_state=0).fit(X, ["happy"] * 10 + ["sad"] * 10)

    citra = np.zeros((128, 128), np.uint8)
    label, conf = prediksi(citra, palsu)
    assert label in ("happy", "sad") and 0.0 <= conf <= 1.0, (label, conf)

    assert prediksi_frame(np.zeros((240, 320, 3), np.uint8), palsu) is None  # tanpa wajah

    _cache.clear()                                  # cache benar-benar menyimpan?
    import joblib as jl, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        jl.dump(palsu, f.name)
        assert muat(f.name) is muat(f.name), "model harus dimuat sekali saja"
        os.unlink(f.name)
    print("classifier OK, confidence=%.3f" % conf)
