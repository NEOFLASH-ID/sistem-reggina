"""Entrypoint Streamlit — sistem pengenalan emosi wajah (HOG + SVM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # agar `src` terbaca

import streamlit as st                                            # noqa: E402
from streamlit.components.v1 import html as components_html  # noqa: E402

from app import page_admin, page_realtime                         # noqa: E402

st.set_page_config(page_title="Deteksi Emosi Wajah", page_icon="🙂", layout="centered")
# Chrome menerjemahkan halaman otomatis dan mengubah DOM, sehingga React Streamlit
# gagal ("NotFoundError: removeChild"). Tandai halaman agar tidak diterjemahkan.
components_html("""<script>
  const d = window.parent.document;
  d.documentElement.setAttribute("translate", "no");
  d.documentElement.classList.add("notranslate");
  if (!d.querySelector('meta[name="google"]')) {
    const m = d.createElement("meta");
    m.name = "google"; m.content = "notranslate";
    d.head.appendChild(m);
  }
</script>""", height=0)

st.sidebar.title("Smart Camera FER")
halaman = st.sidebar.radio("Halaman", ["Deteksi Real-Time", "Admin"])
st.sidebar.caption("HOG + SVM · edge computing · offline")

(page_realtime if halaman == "Deteksi Real-Time" else page_admin).render()
