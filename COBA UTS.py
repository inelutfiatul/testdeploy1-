import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Emotion Detection App", page_icon="🧠", layout="wide")

# ==========================
# HEADER
# ==========================
col1, col2 = st.columns([1, 5])

with col1:
    logo_path = "logo_univ.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown("🎓 **Big Data Project 2025**")

with col2:
    st.title("🧠 Aplikasi Deteksi Ekspresi Wajah")
    st.caption("Dibuat untuk tugas mata kuliah **Big Data** – Mahasiswa Statistika")

st.markdown("---")

# ==========================
# LABEL EKSPRESI
# ==========================
emotion_labels = ['Jijik 🤢', 'Senang 😄', 'Marah 😡', 'Sedih 😢', 'Takut 😱']

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar Wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="📷 Gambar Wajah yang Diupload", use_container_width=True)

    # ==========================
    # PREPROCESSING
    # ==========================
    st.markdown("### 🔄 Tahapan Preprocessing")
    st.write("Gambar akan diubah ukurannya menjadi **48x48 piksel** dan dinormalisasi ke rentang [0,1].")

    img_resized = img.resize((48, 48))  # Sesuaikan dengan input model kamu
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Bentuk (1, 48, 48, 3)

    # ==========================
    # LOAD MODEL
    # ==========================
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("model_ekspresi.h5")

    model = load_model()

    # ==========================
    # PREDIKSI
    # ==========================
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = emotion_labels[predicted_index]
    confidence = np.max(predictions)

    # ==========================
    # HASIL
    # ==========================
    st.markdown("## 🧠 Hasil Analisis Ekspresi Wajah")

    st.image(
        img,
        caption=f"🧩 Ekspresi Terdeteksi: {predicted_label} (Probabilitas: {confidence:.2f})",
        use_container_width=True
    )

    # Menampilkan hasil dengan gaya interaktif
    st.success(f"✅ Model mengenali ekspresi wajah ini sebagai **{predicted_label}** dengan tingkat keyakinan **{confidence:.2%}**.")

    # Tambahkan penjelasan hasil
    st.markdown("---")
    st.markdown("### 📘 Penjelasan Singkat:")
    st.write("""
    - **Model ini menggunakan CNN (Convolutional Neural Network)** untuk menganalisis ekspresi wajah.
    - Gambar wajah yang diunggah akan diklasifikasikan menjadi salah satu dari 5 kategori utama:
      1. Jijik 🤢  
      2. Senang 😄  
      3. Marah 😡  
      4. Sedih 😢  
      5. Takut 😱  
    - Semakin tinggi nilai probabilitas, semakin yakin model terhadap prediksinya.
    """)

else:
    st.info("📥 Silakan unggah gambar wajah terlebih dahulu untuk memulai analisis.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("✨ Dibuat oleh **Ine Lutfiatul Hanifah** | Statistika | Big Data Project 2025")
