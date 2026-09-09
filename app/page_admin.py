"""Halaman Peneliti/Admin: dataset, training, evaluasi, log."""
import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import analysis, classifier, dataset_manager as dm, db, evaluation, logger, training

SPLIT = dm.ROOT / "data" / "split_uji.npz"


@st.cache_data(show_spinner="Menghitung metrik evaluasi...")
def _metrik(mtime_model, mtime_split):
    """Evaluasi ulang model tersimpan pada data uji yang sama. Argumen mtime hanya
    dipakai sebagai kunci cache: berubah kalau model atau data uji diperbarui."""
    d = np.load(SPLIT, allow_pickle=True)
    h = evaluation.evaluasi(classifier.muat(), d["X"], d["y"])
    # PNG tetap diperbarui untuk lampiran laporan; UI menggambar sendiri dari angkanya
    h["gambar_cm"] = str(evaluation.simpan_confusion_matrix(h))
    h["dihitung"] = dt.datetime.now().isoformat(" ", "seconds")
    return h


def _tampilkan_metrik(h):
    st.metric("Akurasi keseluruhan", f"{h['akurasi']:.4f}", f"{h['akurasi'] * 100:.2f}%")
    tabel = pd.DataFrame(h["per_kelas"]).T[["precision", "recall", "f1-score", "support"]]
    tabel.loc["rata-rata"] = [h["rata_rata"][k] for k in
                              ("precision", "recall", "f1-score", "support")]
    st.dataframe(tabel.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                     "f1-score": "{:.4f}", "support": "{:.0f}"}),
                 width="stretch")

    st.caption(f"Confusion matrix — dihitung {h['dihitung']} dari model tersimpan "
               f"terhadap {int(h['confusion_matrix'].sum())} citra uji")
    panjang = pd.DataFrame(
        [{"Aktual": a, "Prediksi": pr, "jumlah": int(n)}
         for i, a in enumerate(h["labels"])
         for pr, n in zip(h["labels"], h["confusion_matrix"][i])])
    dasar = alt.Chart(panjang).encode(
        x=alt.X("Prediksi:N", sort=h["labels"], title="Hasil prediksi"),
        y=alt.Y("Aktual:N", sort=h["labels"], title="Label sebenarnya"))
    st.altair_chart(
        dasar.mark_rect().encode(
            color=alt.Color("jumlah:Q", scale=alt.Scale(scheme="blues"), title="jumlah"),
            tooltip=["Aktual", "Prediksi", "jumlah"])
        + dasar.mark_text(fontSize=16).encode(
            text="jumlah:Q",
            color=alt.condition(alt.datum.jumlah > h["confusion_matrix"].max() / 2,
                                alt.value("white"), alt.value("black"))),
        use_container_width=True)   # altair_chart belum menerima width="stretch"
    with st.expander("Lihat sebagai tabel angka"):
        st.dataframe(pd.DataFrame(h["confusion_matrix"], index=h["labels"],
                                  columns=h["labels"]), width="stretch")
        st.caption(f"Versi gambar untuk lampiran laporan: `{h['gambar_cm']}`")


def render():
    st.header("Administrasi")
    conn = db.connect()
    tab_data, tab_latih, tab_analisis, tab_log = st.tabs(
        ["Dataset", "Training & Evaluasi", "Analisis Emosi", "Log Deteksi"])

    with tab_data:
        ringkas = dm.ringkasan(conn)
        st.dataframe(pd.DataFrame(ringkas) if ringkas else pd.DataFrame(),
                     width="stretch")
        st.caption("Tambah data: taruh citra di `data/raw/<label>/` lalu tekan tombol di bawah.")
        if st.button("Ingest data/raw"):
            st.success(f"Masuk: {dm.ingest(dm.RAW, 'publik', conn)}")
            st.rerun()

    with tab_latih:
        riwayat = [dict(r) for r in conn.execute(
            "SELECT * FROM model ORDER BY id_model DESC")]
        if riwayat:
            st.dataframe(pd.DataFrame(riwayat), width="stretch")
        else:
            st.info("Belum ada model terlatih.")

        st.divider()
        st.subheader("Evaluasi model aktif")
        try:
            _tampilkan_metrik(_metrik(training.MODEL_PATH.stat().st_mtime,
                                      SPLIT.stat().st_mtime))
        except FileNotFoundError:
            st.info("Metrik muncul setelah model dilatih minimal satu kali.")

        st.divider()
        st.warning("Training memakan waktu lama (GridSearchCV). Jangan tutup halaman.")
        if st.button("Latih model sekarang"):
            with st.spinner("Melatih SVM..."):
                X, y = training.muat_fitur(conn)
                model, akurasi, params, X_te, y_te = training.latih(X, y, conn)
                hasil = evaluation.evaluasi(model, X_te, y_te)
            st.success(f"Akurasi uji {akurasi:.4f} — {params}")
            st.code(evaluation.teks(hasil))
            st.image(str(evaluation.simpan_confusion_matrix(hasil)), width="stretch")

    with tab_analisis:
        st.subheader("Analisis Karakteristik Emosional")
        sebar = analysis.distribusi(conn)
        if not sebar:
            st.info("Belum ada hasil deteksi. Gunakan halaman Deteksi Real-Time lebih dulu.")
        else:
            dom = analysis.dominan(conn)
            k1, k2 = st.columns(2)
            k1.metric("Emosi dominan", dom[0], f"{dom[2]:.1f}% dari {sum(b['jumlah'] for b in sebar)} deteksi")
            k2.metric("Rata-rata confidence",
                      f"{sum(b['rata_confidence'] * b['jumlah'] for b in sebar) / sum(b['jumlah'] for b in sebar):.2f}")

            df = pd.DataFrame(sebar).set_index("label_emosi")
            st.bar_chart(df["persentase"], y_label="persentase deteksi (%)")
            st.dataframe(df, width="stretch")

            jam = pd.DataFrame(analysis.per_jam(conn))
            if not jam.empty:
                st.caption("Sebaran deteksi per jam")
                st.bar_chart(jam.pivot_table(index="jam", columns="label_emosi",
                                             values="jumlah", fill_value=0))
            st.write(analysis.naratif(conn))

    with tab_log:
        rekap = logger.rekap(conn)
        if rekap:
            st.dataframe(pd.DataFrame(rekap), width="stretch")
        catatan = logger.baca(conn=conn)
        st.dataframe(pd.DataFrame(catatan) if catatan else pd.DataFrame(),
                     width="stretch")
