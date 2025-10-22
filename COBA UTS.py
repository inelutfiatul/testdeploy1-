import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Digit Detection & Classification", page_icon="🔢", layout="wide")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt")  # Deteksi digit
    classifier = tf.keras.models.load_model("model/INELUTFIATULHANIFAH_LAPORAN 2.h5")  # Klasifikasi ganjil/genap
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# HEADER
# ==========================
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("logo_univ.png"):
        st.image("logo_univ.png", width=90)
    else:
        st.markdown("🎓 **Big Data Project 2025**")

with col2:
    st.title("🔢 Aplikasi Deteksi & Klasifikasi Digit")
    st.caption("Deteksi angka dengan YOLOv8 dan klasifikasi ganjil/genap dengan CNN")

st.markdown("---")

# ==========================
# INPUT GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar digit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # ==========================
    # DETEKSI DENGAN YOLO
    # ==========================
    st.subheader("🔍 Hasil Deteksi Digit")
    results = yolo_model(img)
    result_img = results[0].plot()
    st.image(result_img, caption="📦 Deteksi Digit Menggunakan YOLOv8", use_container_width=True)

    # Jika tidak ada objek terdeteksi
    if len(results[0].boxes) == 0:
        st.warning("⚠️ Tidak ada digit terdeteksi dalam gambar ini.")
    else:
        # ==========================
        # KLASIFIKASI GANJIL/GENAP
        # ==========================
        st.subheader("🧠 Klasifikasi Ganjil / Genap")

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img.crop((x1, y1, x2, y2))  # Potong bagian digit
            cropped_resized = cropped.resize((28, 28))  # Ukuran standar MNIST
            img_array = image.img_to_array(cropped_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)  # 0-9
            confidence = np.max(prediction)

            # Tentukan ganjil/genap
            label = "GENAP" if class_index % 2 == 0 else "GANJIL"

            st.markdown(f"""
            **Digit {i+1}:**
            - Prediksi Angka → **{class_index}**
            - Jenis Angka → 🟢 **{label}**
            - Keyakinan Model → {confidence:.2f}
            """)

        st.success("✅ Semua digit berhasil diklasifikasikan!")

else:
    st.info("📥 Silakan unggah gambar berisi digit untuk memulai deteksi dan klasifikasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("✨ Dibuat oleh **Ine Lutfiatul Hanifah** | Statistika | Big Data Project 2025")
