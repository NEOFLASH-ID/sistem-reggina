"""UC6: Melatih Model SVM dari citra di tabel `dataset` (offline, bukan real-time)."""
import datetime as dt
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from . import db
from .dataset_manager import ROOT
from .feature_extraction import extract

MODEL_PATH = ROOT / "models" / "svm_model.joblib"

# BAB IV 4.3.5: coba linear dan rbf, C/gamma dituning GridSearchCV
GRID = [
    {"kernel": ["linear"], "C": [0.1, 1, 10]},
    {"kernel": ["rbf"], "C": [1, 10, 100], "gamma": ["scale", 0.01, 0.001]},
]


def muat_fitur(conn=None):
    """Baca semua citra dataset -> matriks fitur HOG X dan label y."""
    conn = conn or db.connect()
    X, y = [], []
    for row in conn.execute("SELECT path_file, label_emosi FROM dataset"):
        citra = cv2.imread(str(ROOT / row["path_file"]), cv2.IMREAD_GRAYSCALE)
        if citra is None:
            continue                      # file hilang/rusak: lewati, jangan matikan training
        X.append(extract(citra))
        y.append(row["label_emosi"])
    return np.array(X), np.array(y)


def latih(X, y, conn=None, simpan_ke=MODEL_PATH, cv=5):
    """Split 80:20 stratified -> GridSearchCV -> simpan model + catat akurasi ke tabel `model`.

    Mengembalikan (model terbaik, akurasi data uji, parameter terbaik, X_test, y_test).
    X_test/y_test dikembalikan supaya evaluation.py memakai split yang sama persis.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # ponytail: probability=True bikin SVC melatih ulang lewat Platt scaling (lebih lambat),
    # tapi log_deteksi wajib punya confidence_score, jadi memang dibutuhkan.
    gs = GridSearchCV(SVC(decision_function_shape="ovr", probability=True,
                          random_state=42),
                      GRID, cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    akurasi = gs.score(X_te, y_te)

    if simpan_ke is not None:
        Path(simpan_ke).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(gs.best_estimator_, simpan_ke)
        conn = conn or db.connect()
        # simpan path relatif terhadap folder proyek: isi tabel ikut pindah kalau proyek
        # dipindah, dan tidak membocorkan struktur folder mesin saat ditampilkan ke pengguna
        jalur = Path(simpan_ke)
        jalur = jalur.relative_to(ROOT) if jalur.is_relative_to(ROOT) else jalur
        conn.execute("INSERT INTO model (path_model, akurasi, tanggal_training)"
                     " VALUES (?,?,?)",
                     (str(jalur), float(akurasi), dt.datetime.now().isoformat(" ", "seconds")))
        conn.commit()
    return gs.best_estimator_, akurasi, gs.best_params_, X_te, y_te


if __name__ == "__main__":
    # Data sintetis: 4 kelas terpisah jelas -> kalau pipeline benar, akurasi harus tinggi.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 0.3, (25, 8)) for m in (0, 3, 6, 9)])
    y = np.repeat(["happy", "sad", "angry", "neutral"], 25)

    conn = db.connect(":memory:")
    model, akurasi, params, X_te, y_te = latih(X, y, conn, simpan_ke=None, cv=3)
    assert akurasi > 0.9, akurasi
    assert len(y_te) == 20 and len(set(y_te)) == 4, "split 80:20 harus stratified"
    assert set(model.predict_proba(X_te[:1])[0].round(3)) , "confidence tersedia"
    print(f"training OK, akurasi sintetis={akurasi:.2f}, {params}")
