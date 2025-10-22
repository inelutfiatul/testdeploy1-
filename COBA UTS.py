# ======================================
# IMPORT LIBRARY
# ======================================
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
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

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/INELUTFIATULHANIFAH_LAPORAN 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.caption("© 2025 | Dashboard Klasifikasi AI | Dibuat dengan ❤️ menggunakan Streamlit, YOLOv8 & TensorFlow")
