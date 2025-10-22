import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==========================
# CONFIG & STYLE
# ==========================
st.set_page_config(page_title="Klasifikasi Ekspresi & Digit", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 32px;
        color: #4B7BE5;
        font-weight: bold;
    }
    .subheader {
        color: #333;
        font-size: 20px;
        text-align: center;
        margin-top: -10px;
    }
    .result-box {
        background-color: #F5F7FF;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    face_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    digit_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    # Cek keberadaan file
    if not os.path.exists(face_path):
        st.error("❌ File model ekspresi wajah (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(digit_path):
        st.error("❌ File model digit angka (.h5) tidak ditemukan.")
        st.stop()

    # Load model
    face_model = YOLO(face_path)
    digit_model = tf.keras.models.load_model(digit_path)
    return face_model, digit_model

# ==========================
# UI HEADER
# ==========================
st.markdown("<div class='title'>🧠 Dashboard Klasifikasi Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Proyek UAS – Big Data & AI</div>", unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns([1, 3])
with col1:
    st.sidebar.image("LOGO USK.png", width=150)
with col2:
    st.sidebar.header("⚙️ Pengaturan")

menu = st.sidebar.radio("Pilih Jenis Klasifikasi:", ["Ekspresi Wajah", "Digit Angka"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROCESSING
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # ===================================
    # 1️⃣ EKSPRESI WAJAH (.pt)
    # ===================================
    if menu == "Ekspresi Wajah":
        st.subheader("🔍 Hasil Deteksi Ekspresi Wajah")
        results = face_model(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="📸 Deteksi Ekspresi", use_container_width=True)

        if len(results[0].boxes) == 0:
            st.warning("⚠️ Tidak ada wajah terdeteksi.")
        else:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls].capitalize()

                with st.container():
                    st.markdown(f"<div class='result-box'><h3>Ekspresi: 😄 {label}</h3><p>Keyakinan: {conf:.2f}</p></div>", unsafe_allow_html=True)

    # ===================================
    # 2️⃣ DIGIT ANGKA (.h5)
    # ===================================
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Digit Angka")

        # preprocessing gambar sesuai model digit
        img_gray = img.convert("L")
        img_resized = img_gray.resize((28, 28))
        img_array = np.array(img_resized).reshape(1, 28, 28, 1)
        img_array = img_array.astype('float32') / 255.0

        pred = digit_model.predict(img_array)
        pred_label = int(np.argmax(pred))
        prob = float(np.max(pred))

        # tampilkan hasil
        colA, colB = st.columns(2)
        with colA:
            st.image(img_gray, caption="🖼️ Gambar Uji", use_container_width=True)
        with colB:
            parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"
            st.markdown(f"""
                <div class='result-box'>
                    <h2>Angka: {pred_label}</h2>
                    <h4>Akurasi: {prob:.2%}</h4>
                    <p>{parity}</p>
                </div>
            """, unsafe_allow_html=True)

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk melakukan deteksi atau klasifikasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 – Dibuat oleh Ine Lutfia • Proyek UAS Big Data</p>", unsafe_allow_html=True)
