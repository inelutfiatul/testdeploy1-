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
        st.subheader("😄 Hasil Klasifikasi Ekspresi Wajah")

        results = face_model(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="🔍 Deteksi Ekspresi", use_container_width=True)

        if len(results[0].boxes) == 0:
            st.warning("⚠️ Tidak ada wajah terdeteksi.")
        else:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls]
                st.success(f"✅ Ekspresi: **{label}** ({conf:.2f})")
# -------------------------------------------------
    # 2️⃣ KLASIFIKASI DIGIT ANGKA (.H5 / KERAS)
    # -------------------------------------------------
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Digit Angka")

        # Pra-pemrosesan gambar untuk model .h5
        img_resized = img.resize((28, 28)).convert("L")   # ubah ke grayscale
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)     # (1, 28, 28, 1)
        img_array = img_array / 255.0


        pred = digit_model.predict(img_array)
        predicted_label = np.argmax(pred)
        confidence = np.max(pred)

        st.image(img_resized, caption="🧩 Gambar Setelah Dikonversi (28x28 Grayscale)", width=150)
        st.success(f"✅ Prediksi Angka: **{predicted_label}** (Keyakinan: {confidence:.2f})")

else:
    st.info("📥 Silakan unggah gambar terlebih dahulu.")

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.caption("© 2025 | Dashboard Klasifikasi AI | Dibuat dengan ❤️ menggunakan Streamlit, YOLOv8 & TensorFlow")
