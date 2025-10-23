import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="UTS Dashboard – Ine Lutfia", page_icon="🤖", layout="wide")

# ==== STYLE DASHBOARD ====
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #E0EAFC, #CFDEF3);
        font-family: 'Poppins', sans-serif;
    }
    .title { text-align:center; font-size:40px; color:#2F4F9D; font-weight:bold; }
    .subtitle { text-align:center; font-size:18px; color:gray; margin-top:-10px; }
    .result-box {
        background-color:#F8FAFF; padding:20px; border-radius:15px; text-align:center;
        box-shadow:0 2px 15px rgba(0,0,0,0.1); margin-top:20px;
    }
    .emoji-rain {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        overflow: hidden;
        z-index: -1;
        animation: fall 10s linear infinite;
    }
    @keyframes fall {
        0% { transform: translateY(-10%); }
        100% { transform: translateY(100vh); }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# STATE NAVIGASI
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Cover"

def goto(page):
    st.session_state.page = page

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    face_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    digit_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    if not os.path.exists(face_path):
        st.error("❌ Model ekspresi (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(digit_path):
        st.error("❌ Model digit (.h5) tidak ditemukan.")
        st.stop()

    face_model = YOLO(face_path)
    digit_model = tf.keras.models.load_model(digit_path)
    return face_model, digit_model

face_model, digit_model = load_models()

# ==========================
# HALAMAN COVER
# ==========================
if st.session_state.page == "Cover":
    st.markdown("<div class='title'>🎓 Dashboard UTS – Big Data & AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Dibuat oleh <b>Ine Lutfia</b> | Universitas Syiah Kuala</div>", unsafe_allow_html=True)
    st.image("LOGO USK.png", width=200)
    st.markdown("<div class='emoji-rain'>✨ ✨ ✨ ✨</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("😃 Deteksi Ekspresi Wajah"):
            goto("Face Detection")
    with col2:
        if st.button("🔢 Klasifikasi Angka"):
            goto("Digit Classifier")

# ==========================
# HALAMAN DETEKSI EKSPRESI
# ==========================
elif st.session_state.page == "Face Detection":
    st.markdown("<h2>🧠 Deteksi Ekspresi Wajah</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📸 Unggah gambar wajah Anda", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar Input", use_container_width=True)

        try:
            results = face_model(img)
            annotated = results[0].plot()
            st.image(annotated, caption="📸 Hasil Deteksi", use_container_width=True)

            if len(results[0].boxes) == 0:
                st.warning("⚠️ Tidak ada wajah terdeteksi.")
            else:
                labels = results[0].names
                best_conf, best_label = 0, "tidak dikenali"
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_label = labels[int(box.cls[0])].lower()

                # Motivasi sesuai ekspresi
                motivasi = {
                    "senang": "Energi positifmu menular, tetap tersenyum ya! 🌞",
                    "bahagia": "Kamu bahagia banget hari ini, semoga selalu begitu! 💖",
                    "sedih": "Semua akan baik-baik saja, percayalah 🌧️",
                    "marah": "Tenangkan hati dulu, kamu lebih hebat dari emosimu 💪",
                    "takut": "Rasa takut itu wajar, tapi kamu berani menghadapi! 🌿",
                    "jijik": "Wajar merasa begitu, tapi kamu tetap keren kok 😅"
                }.get(best_label, "Ekspresimu unik banget! 🌈")

                emoji = {
                    "senang": "😄", "bahagia": "😊", "sedih": "😢",
                    "marah": "😡", "takut": "😱", "jijik": "🤢"
                }.get(best_label, "🙂")

                st.markdown(f"""
                    <div class='result-box'>
                        <h2>{emoji} Ekspresi: <b>{best_label.capitalize()}</b></h2>
                        <p>{motivasi}</p>
                        <p>🎯 Akurasi: {best_conf*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat deteksi ekspresi: {e}")

    st.button("➡️ Lanjut ke Klasifikasi Angka", on_click=lambda: goto("Digit Classifier"))
    st.button("⬅️ Kembali ke Cover", on_click=lambda: goto("Cover"))

# ==========================
# HALAMAN KLASIFIKASI ANGKA
# ==========================
elif st.session_state.page == "Digit Classifier":
    st.markdown("<h2>🔢 Klasifikasi Angka Tulisan Tangan</h2>", unsafe_allow_html=True)
    uploaded_digit = st.file_uploader("📸 Upload gambar angka tulisan tangan", type=["jpg", "jpeg", "png"])

    if uploaded_digit is not None:
        img = Image.open(uploaded_digit).convert('L')
        img = img.resize((28, 28))
        arr = np.array(img).astype("float32") / 255.0

        try:
            arr = np.expand_dims(arr, axis=(0, -1))
            pred = digit_model.predict(arr)
            angka = int(np.argmax(pred))
            prob = float(np.max(pred))
            parity = "✅ GENAP" if angka % 2 == 0 else "⚠️ GANJIL"

            st.image(img, caption="🖼️ Gambar (Preprocessed)", width=150)
            st.markdown(f"""
                <div class='result-box'>
                    <h2>Angka: {angka}</h2>
                    <h4>Akurasi: {prob*100:.2f}%</h4>
                    <p>{parity}</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"🚨 Terjadi kesalahan prediksi: {e}")

    else:
        st.info("⬆️ Upload gambar angka untuk klasifikasi")

    st.button("⬅️ Kembali ke Ekspresi", on_click=lambda: goto("Face Detection"))
    st.button("➡️ Tentang", on_click=lambda: goto("About"))

# ==========================
# HALAMAN ABOUT
# ==========================
elif st.session_state.page == "About":
    st.markdown("<h2>🤖 Tentang Aplikasi AI-ku</h2>", unsafe_allow_html=True)
    st.markdown("""
        Dashboard ini dibuat sebagai proyek **Ujian Tengah Semester (UTS)** untuk mata kuliah **Big Data & Artificial Intelligence**.  
        Aplikasi ini menggabungkan dua model AI:
        - 🧠 *Deteksi Ekspresi Wajah* (YOLOv8)
        - 🔢 *Klasifikasi Angka Tulisan Tangan* (CNN TensorFlow)

        🌟 Fitur unggulan:
        - Desain interaktif & smooth transition  
        - Pesan motivasi otomatis sesuai ekspresi  
        - Tampilan mirip slide presentasi  
    """)
    st.button("⬅️ Kembali ke Cover", on_click=lambda: goto("Cover"))
