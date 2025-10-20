import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt")
    classifier = tf.keras.models.load_model("model/INELUTFIATULHANIFAH_LAPORAN 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Tampilan Header
# ==========================
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo_univ.png", width=90)  # Ganti dengan nama file logo kamu
with col2:
    st.markdown("""
    ### 💻 Aplikasi Deteksi & Klasifikasi Gambar  
    **oleh Ine Lutfiatul Hanifah**  
    *Proyek Big Data - 2025*
    """)

st.markdown("---")

# ==========================
# Pilihan Mode
# ==========================
menu = st.sidebar.selectbox("🎯 Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📁 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Caption otomatis sesuai mode
    if menu == "Deteksi Objek (YOLO)":
        caption_input = "📸 Gambar yang akan dideteksi menggunakan YOLOv8"
    else:
        caption_input = "🖼️ Gambar yang akan diklasifikasikan menggunakan CNN"

    st.image(img, caption=caption_input, use_container_width=True)
    st.markdown("---")

    # ==========================
    # Mode: Deteksi Objek
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        with st.spinner("🔍 Sedang melakukan deteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="✅ Hasil Deteksi Objek (YOLOv8)", use_container_width=True)
            st.success("Deteksi objek selesai!")

    # ==========================
    # Mode: Klasifikasi Gambar
    # ==========================
    elif menu == "Klasifikasi Gambar":
        with st.spinner("🧠 Sedang melakukan klasifikasi gambar..."):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Label opsional
            label_dict = {
                0: "Kelas 0",
                1: "Kelas 1",
                2: "Kelas 2"
            }
            label = label_dict.get(class_index, f"Kelas {class_index}")

            st.image(img, caption=f"🧩 Hasil Klasifikasi: {label} (Probabilitas: {confidence:.2f})", use_container_width=True)
            st.success("Klasifikasi gambar selesai!")

    # ==========================
    # Watermark
    # ==========================
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>© 2025 Ine Lutfiatul Hanifah | Big Data Project</p>", unsafe_allow_html=True)
