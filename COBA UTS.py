# ======================================
# IMPORT LIBRARY
# ======================================
import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# ======================================
# KONFIGURASI DASAR
# ======================================
st.set_page_config(
    page_title="Dashboard Klasifikasi AI",
    page_icon="🧠",
    layout="centered"
)

# ======================================
# HEADER
# ======================================
col1, col2 = st.columns([1, 4])

with col1:
    logo_path = "LOGO USK.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown("🎓 **Big Data Project 2025**")

with col2:
    st.title("Aplikasi Klasifikasi Gambar AI")
    st.markdown("Klasifikasi **Ekspresi Wajah (YOLOv8)** & **Digit Angka (Keras)**")

st.markdown("---")

# ======================================
# LOAD MODEL
# ======================================
@st.cache_resource
def load_models():
    if not os.path.exists("Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"):
        st.error("❌ File model ekspresi_wajah.pt tidak ditemukan!")
        st.stop()
    if not os.path.exists("INELUTFIATULHANIFAH_LAPORAN 2.h5"):
        st.error("❌ File model digit_model.h5 tidak ditemukan!")
        st.stop()

    face_model = YOLO("Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt")
    digit_model = tf.keras.models.load_model("INELUTFIATULHANIFAH_LAPORAN 2.h5")
    return face_model, digit_model

face_model, digit_model = load_models()

# ======================================
# MENU
# ======================================
menu = st.sidebar.radio("📌 Pilih Jenis Klasifikasi:", ["Ekspresi Wajah", "Digit Angka"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ======================================
# PROSES GAMBAR
# ======================================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # -------------------------------------------------
    # 1️⃣ KLASIFIKASI EKSPRESI WAJAH (YOLO)
    # -------------------------------------------------
    if menu == "Ekspresi Wajah":
        st.subheader("😄
