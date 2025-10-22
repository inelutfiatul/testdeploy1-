import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import torch
import os

# ==========================
# CONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Dashboard AI Klasifikasi", page_icon="🤖", layout="wide")

col1, col2 = st.columns([1, 5])

# Logo
with col1:
    logo_path = "LOGO USK.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown("🎓 **Big Data Project 2025**")

with col2:
    st.title("🤖 Dashboard Klasifikasi Gambar")
    st.caption("Dibuat oleh **Ine Lutfiatul Hanifah** | Statistika | Big Data Project 2025")

st.markdown("---")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    face_model = YOLO("Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt")  # Model ekspresi wajah
    digit_model = tf.keras.models.load_model("INELUTFIATULHANIFAH_LAPORAN 2.h5")  # Model digit angka
    return face_model, digit_model

face_model, digit_model = load_models()

# ==========================
# MENU PILIHAN
# ==========================
menu = st.sidebar.radio("Pilih Jenis Klasifikasi:", ["Ekspresi Wajah", "Digit Angka"])

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES GAMBAR
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # ==========================
    # 1️⃣ KLASIFIKASI EKSPRESI WAJAH
    # ==========================
    if menu == "Ekspresi Wajah":
        st.subheader("🧠 Hasil Klasifikasi Ekspresi Wajah")

        results = face_model(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="🔍 Hasil Deteksi Ekspresi Wajah", use_container_width=True)

        # Ambil hasil prediksi (kalau model tidak mendeteksi)
        if len(results[0].boxes) == 0:
            st.warning("⚠️ Tidak ada wajah terdeteksi pada gambar ini.")
        else:
            # Ambil label kelas dan confidence tertinggi
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls]
                st.success(f"✅ Teridentifikasi ekspresi: **{label}** (keyakinan: {conf:.2f})")

    # ==========================
    # 2️⃣ KLASIFIKASI DIGIT ANGKA
    # ==========================
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Angka")

        # Preprocessing untuk model digit
        img_gray = img.convert("L").resize((28, 28))  # Asumsi model MNIST 28x28 grayscale
        img_array = image.img_to_array(img_gray)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi model
        prediction = digit_model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Tentukan kategori ganjil/genap
        kategori = "Genap" if class_index % 2 == 0 else "Ganjil"

        st.image(img, caption=f"🧠 Prediksi Angka: {class_index}", use_container_width=True)
        st.info(f"📊 Angka Terdeteksi: **{class_index}** | Keyakinan: **{confidence:.2f}** | Kategori: **{kategori}**")

else:
    st.info("📥 Silakan unggah gambar terlebih dahulu untuk memulai klasifikasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("✨ Dibuat oleh **Ine Lutfiatul Hanifah** | Statistika | Big Data Project 2025")
