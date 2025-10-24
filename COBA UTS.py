import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import time

# ==========================
# 🌈 CONFIG DASAR
# ==========================
st.set_page_config(page_title="AI Dashboard – Ine Lutfia", page_icon="🤖", layout="wide")

# ==========================
# 🌌 ANIMASI & STYLING
# ==========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

body {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #4a148c);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    font-family: 'Poppins', sans-serif;
    color: #E0EAF5;
}

@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.fade-in {
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}

.glass-card {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(31,38,135,0.3);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease-in-out;
}
.glass-card:hover {
    box-shadow: 0 0 30px rgba(0,255,255,0.5);
    transform: scale(1.02);
}

.neon {
    color: #00FFFF;
    text-shadow: 0px 0px 10px #00FFFF, 0px 0px 20px #00FFFF;
}
.footer {
    text-align: center;
    color: #B0E0E6;
    margin-top: 30px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# 🧠 LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    face_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    digit_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    if not os.path.exists(face_path):
        st.error("❌ Model ekspresi wajah (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(digit_path):
        st.error("❌ Model digit angka (.h5) tidak ditemukan.")
        st.stop()

    face_model = YOLO(face_path)
    digit_model = tf.keras.models.load_model(digit_path)
    return face_model, digit_model

face_model, digit_model = load_models()

# ==========================
# 🧭 NAVIGASI MULTI-HALAMAN
# ==========================
page = st.sidebar.radio("📂 Navigasi Halaman", ["🏠 Cover", "😄 Ekspresi Wajah", "🔢 Klasifikasi Angka", "ℹ️ Tentang"])
st.sidebar.markdown("---")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=150)

# ==========================
# 🏠 HALAMAN COVER
# ==========================
if page == "🏠 Cover":
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;' class='neon'>🤖 UTS Dashboard – Ine Lutfia</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Big Data & Artificial Intelligence | Universitas Syiah Kuala</p>", unsafe_allow_html=True)
    st.image("LOGO USK.png", width=220)
    st.markdown("""
        <div class='glass-card fade-in' style='text-align:center;'>
            <p>Selamat datang di dashboard AI! 🌟</p>
            <p>Aplikasi ini menggabungkan dua kecerdasan buatan:
            <br>🧠 <b>Deteksi Ekspresi Wajah</b> &nbsp; dan &nbsp; 🔢 <b>Klasifikasi Angka Tulisan Tangan</b></p>
            <p>Klik menu di samping untuk memulai 🚀</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# 😄 DETEKSI EKSPRESI WAJAH
# ==========================
elif page == "😄 Ekspresi Wajah":
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.header("🎭 Deteksi Ekspresi Wajah")
    uploaded_face = st.file_uploader("📸 Unggah gambar wajah", type=["jpg", "jpeg", "png"])

    if uploaded_face:
        img = Image.open(uploaded_face).convert("RGB")
        st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

        try:
            results = face_model(img)
            annotated = results[0].plot()
            st.image(annotated, caption="📸 Hasil Deteksi", use_container_width=True)

            if len(results[0].boxes) == 0:
                st.warning("😅 Tidak ada wajah terdeteksi. Pastikan wajah terlihat jelas & cukup terang.")
            else:
                labels = results[0].names
                best_conf, best_label = 0, "tidak dikenali"
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_label = labels[int(box.cls[0])].lower()

                emoji = {"senang": "😄", "bahagia": "😊", "sedih": "😢", "marah": "😡", "takut": "😱", "jijik": "🤢"}.get(best_label, "🙂")
                motivasi = {
                    "senang": "Energi positifmu luar biasa! 🌞",
                    "bahagia": "Terus tebarkan senyummu ya! 💖",
                    "sedih": "Semua akan baik-baik saja, aku yakin kamu kuat 💪",
                    "marah": "Ambil napas dulu, kamu hebat kok 😌",
                    "takut": "Rasa takut itu wajar, tapi kamu pemberani 🌿",
                    "jijik": "Tetap tenang, semua aman kok 😅"
                }.get(best_label, "Ekspresimu unik banget! 🌈")

                st.markdown(f"""
                    <div class='glass-card'>
                        <h2 class='neon'>{emoji} Ekspresi: {best_label.capitalize()}</h2>
                        <p>{motivasi}</p>
                        <p>Akurasi Deteksi: <b>{best_conf*100:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
    else:
        st.info("⬆️ Silakan upload gambar wajah terlebih dahulu.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# 🔢 KLASIFIKASI ANGKA
# ==========================
elif page == "🔢 Klasifikasi Angka":
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.header("🔢 Klasifikasi Angka Tulisan Tangan")
    uploaded_digit = st.file_uploader("📤 Unggah gambar angka (tulisan tangan)", type=["jpg", "jpeg", "png"])

    if uploaded_digit:
        img = Image.open(uploaded_digit)
        st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

        try:
            input_shape = digit_model.input_shape
            height, width = input_shape[1], input_shape[2]
            channels = input_shape[3] if len(input_shape) == 4 else 3

            # Auto konversi channel
            if channels == 1:
                img = img.convert("L")
            else:
                img = img.convert("RGB")

            img = img.resize((width, height))
            arr = np.array(img).astype("float32") / 255.0

            if channels == 1 and arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            elif channels == 3 and arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)

            arr = np.expand_dims(arr, axis=0)

            pred = digit_model.predict(arr)
            angka = int(np.argmax(pred))
            prob = float(np.max(pred))
            parity = "✅ GENAP" if angka % 2 == 0 else "⚠️ GANJIL"

            st.markdown(f"""
                <div class='glass-card'>
                    <h2 class='neon'>Angka Terdeteksi: {angka}</h2>
                    <p>Akurasi: <b>{prob*100:.2f}%</b></p>
                    <p>{parity}</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"🚨 Terjadi kesalahan prediksi: {e}")
    else:
        st.info("⬆️ Upload gambar angka tulisan tangan terlebih dahulu.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# ℹ️ HALAMAN TENTANG
# ==========================
elif page == "ℹ️ Tentang":
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    <div class='glass-card'>
        <p><b>AI Dashboard</b> ini merupakan proyek <b>Ujian Tengah Semester</b> mata kuliah <b>Big Data & Artificial Intelligence</b>.</p>
        <ul>
            <li>🧠 Model 1: Deteksi Ekspresi Wajah (YOLOv8)</li>
            <li>🔢 Model 2: Klasifikasi Angka Tulisan Tangan (CNN TensorFlow)</li>
        </ul>
        <p>Dibuat oleh <b>Ine Lutfia</b> – Universitas Syiah Kuala 💙</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# 🌙 FOOTER
# ==========================
st.markdown("<div class='footer'>© 2025 – Ine Lutfia | UTS Big Data & AI ✨</div>", unsafe_allow_html=True)
