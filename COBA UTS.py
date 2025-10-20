import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Ine Lutfiatul Hanifah_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/INELUTFIATULHANIFAH_LAPORAN 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# HEADER / UI AWAL
# ==========================
st.set_page_config(page_title="Big Data App", page_icon="🧠", layout="wide")

col1, col2 = st.columns([1, 5])

# Logo dengan proteksi error
with col1:
    logo_path = "logo_univ.png"  # ubah sesuai nama file kamu
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown("🎓 **Big Data Project 2025**")

with col2:
    st.title("🧠 Aplikasi Deteksi & Klasifikasi Gambar")
    st.caption("Dibuat untuk tugas mata kuliah **Big Data** – Mahasiswa Statistika")

st.markdown("---")

# ==========================
# MENU PILIHAN
# ==========================
menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES GAMBAR
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar Input dari Pengguna", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()

        st.image(result_img, caption="🔍 Hasil Deteksi Objek menggunakan YOLOv8", use_container_width=True)

        # Cek apakah objek terdeteksi
        if len(results[0].boxes) == 0:
            st.warning("⚠️ Tidak ada objek terdeteksi pada gambar ini.")
        else:
            st.success(f"✅ {len(results[0].boxes)} objek berhasil terdeteksi!")

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.image(img, caption=f"🧠 Hasil Klasifikasi Gambar (Kelas: {class_index}, Probabilitas: {confidence:.2f})",
                 use_container_width=True)

        st.info(f"🧾 Prediksi kelas: **{class_index}** dengan tingkat keyakinan **{confidence:.2f}**")

else:
    st.info("📥 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("✨ Dibuat oleh **Ine Lutfiatul Hanifah** | Statistika | Big Data Project 2025")
