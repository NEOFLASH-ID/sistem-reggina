"""UC5: halaman deteksi real-time untuk Pengguna."""
import cv2
import numpy as np
import streamlit as st

from src import capture, classifier, logger
from src.face_detection import crop, detect

NAMA = {"happy": "Senang 😊", "sad": "Sedih 😢", "angry": "Marah 😠", "neutral": "Netral 😐"}


def _gambar(frame, box, label, conf):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(frame, f"{label} {conf:.0%}", (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render():
    st.header("Deteksi Emosi Real-Time")
    # Jepret didahulukan: memakai kamera browser pengunjung, jadi ini satu-satunya mode
    # yang berguna saat aplikasi dibuka dari jarak jauh (mis. lewat tunnel publik).
    mode = st.radio("Mode kamera", ["Jepret", "Live"], horizontal=True,
                    key="kamera_mode",
                    captions=["Pakai kamera perangkat Anda — pilih ini kalau membuka dari jauh",
                              "Video dari kamera komputer server, bukan kamera Anda"])
    try:
        model = classifier.muat()
    except FileNotFoundError:
        st.error("Model belum ada. Latih dulu di halaman Admin.")
        return

    if mode == "Jepret":
        foto = st.camera_input("Kamera", label_visibility="collapsed")
        if foto is None:
            return
        frame = cv2.imdecode(np.frombuffer(foto.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        box = detect(frame)
        if box is None:
            st.warning("Wajah tidak terdeteksi — dekatkan wajah, pastikan cukup terang.")
            return
        label, conf = classifier.prediksi(crop(frame, box), model)
        logger.catat(label, conf)
        st.image(_gambar(frame, box, NAMA.get(label, label), conf), width="stretch")
        st.metric(NAMA.get(label, label), f"{conf:.1%} yakin")
        return

    if not st.toggle("Nyalakan kamera"):
        st.info("Nyalakan untuk mulai. Matikan toggle untuk berhenti.")
        return

    layar, papan = st.empty(), st.empty()
    label = conf = sebelumnya = None
    try:
        for frame in capture.frames():
            box = detect(frame)
            if box is not None:
                label, conf = classifier.prediksi(crop(frame, box), model)
                if label != sebelumnya:      # log saat emosi berubah saja
                    logger.catat(label, conf)
                    sebelumnya = label
                if label:
                    frame_rgb = _gambar(frame, box, NAMA.get(label, label), conf)
                    papan.metric(NAMA.get(label, label), f"{conf:.1%} yakin")
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            layar.image(frame_rgb, width="stretch")
    except RuntimeError as e:
        st.error(str(e))
