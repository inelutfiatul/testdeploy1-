import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import tensorflow as tf
import os

# ==============================
# CONFIGURASI AWAL
# ==============================
st.set_page_config(page_title="AI Dashboard UTS", page_icon="🤖", layout="wide")

# ===== CSS FUTURISTIK =====
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 20% 30%, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    text-align: center;
    color: #00eaff;
    text-shadow: 0px 0px 15px #00eaff;
}
.neon-box {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 30px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 15px rgba(0,255,255,0.3);
}
.navbar {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 40px;
}
.nav-button {
    background: rgba(0, 238, 255, 0.15);
    border: 1px solid #00eaff;
    border-radius: 12px;
    padding: 10px 18px;
    color: #00eaff;
    font-weight: bold;
    cursor: pointer;
    transition: 0.3s;
}
.nav-button:hover {
    background: #00eaff;
    color: black;
}
.result-box {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_models():
    face_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    digit_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    if not os.path.exists(face_path):
        st.error("❌ File model ekspresi wajah (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(digit_path):
        st.error("❌ File model digit angka (.h5) tidak ditemukan.")
        st.stop()

    face_model = YOLO(face_path)
    digit_model = tf.keras.models.load_model(digit_path)
    return face_model, digit_model

face_model, digit_model = load_models()

# ==============================
# NAVIGASI SLIDE
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "Cover"

def goto(page_name):
    st.session_state.page = page_name

st.markdown("<div class='navbar'>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Cover", key="cover"): goto("Cover")
with col2:
    if st.button("😄 Ekspresi Wajah", key="face"): goto("Face Detection")
with col3:
    if st.button("🔢 Angka", key="digit"): goto("Digit Classifier")
with col4:
    if st.button("💡 Tentang", key="about"): goto("About")
st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# HALAMAN 1 — COVER
# ==============================
if st.session_state.page == "Cover":
    st.markdown("""
        <div class="neon-box">
            <h1>🤖 DASHBOARD AI UTS</h1>
            <h3>Deteksi Ekspresi Wajah & Klasifikasi Angka</h3>
            <p style='text-align:center; color:#ccc;'>
                Dibuat oleh <b>Ine Lutfia</b><br>
                Proyek ini menggunakan Deep Learning untuk mengenali ekspresi manusia dan angka tulisan tangan.<br>
                Nikmati pengalaman interaktif dan desain futuristik! 🚀
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("➡️ Mulai", on_click=lambda: goto("Face Detection"))

# ==============================
# HALAMAN 2 — DETEKSI EKSPRESI WAJAH
# ==============================
elif st.session_state.page == "Face Detection":
    st.markdown("<h2>😄 Deteksi Ekspresi Wajah</h2>", unsafe_allow_html=True)
    uploaded_face = st.file_uploader("📸 Upload gambar wajah", type=["jpg", "jpeg", "png"])

    if uploaded_face is not None:
        img = Image.open(uploaded_face).convert("RGB")
        st.image(img, caption="Gambar asli", use_container_width=True)

        results = face_model(img)
        annotated = results[0].plot()
        st.image(annotated, caption="✨ Hasil Deteksi Wajah", use_container_width=True)

        if len(results[0].boxes) == 0:
            st.warning("🚫 Tidak ada wajah terdeteksi.")
        else:
            model_labels = results[0].names
            emoji_map = {
                "senang": "😄", "bahagia": "😊", "sedih": "😢",
                "marah": "😡", "takut": "😱", "jijik": "🤢"
            }

            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model_labels.get(cls, "Tidak dikenal").lower()
                emoji = emoji_map.get(label, "🙂")

                st.markdown(f"""
                    <div class='result-box'>
                        <h3>{emoji} Ekspresi: <b>{label.capitalize()}</b></h3>
                        <p>🎯 Keyakinan: <b>{conf*100:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

    else:
        st.info("⬆️ Upload foto wajah terlebih dahulu")

    st.button("⬅️ Kembali", on_click=lambda: goto("Cover"))
    st.button("➡️ Lanjut ke Klasifikasi Angka", on_click=lambda: goto("Digit Classifier"))

# ==============================
# HALAMAN 3 — KLASIFIKASI ANGKA
# ==============================
elif st.session_state.page == "Digit Classifier":
    st.markdown("<h2>🔢 Klasifikasi Angka</h2>", unsafe_allow_html=True)
    uploaded_digit = st.file_uploader("📸 Upload gambar angka tulisan tangan", type=["jpg", "jpeg", "png"])

    if uploaded_digit is not None:
        img = Image.open(uploaded_digit).convert('L')
        img = img.resize((28, 28))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))

        pred = digit_model.predict(arr)
        angka = np.argmax(pred)
        prob = np.max(pred)
        parity = "✅ GENAP" if angka % 2 == 0 else "⚠️ GANJIL"

        st.image(img, caption="🖼️ Gambar preprocessed", width=150)
        st.markdown(f"""
            <div class='result-box'>
                <h2>Angka: {angka}</h2>
                <h4>Akurasi: {prob*100:.2f}%</h4>
                <p>{parity}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⬆️ Upload gambar angka untuk diklasifikasi")

    st.button("⬅️ Kembali ke Ekspresi", on_click=lambda: goto("Face Detection"))
    st.button("➡️ Tentang AI-ku", on_click=lambda: goto("About"))

# ==============================
# HALAMAN 4 — TENTANG AI-KU
# ==============================
elif st.session_state.page == "About":
    st.markdown("""
        <div class="neon-box">
            <h2>💡 Tentang AI-ku</h2>
            <p style='text-align:center; color:#ccc;'>
                Dashboard ini menggunakan dua model deep learning:
                <ul>
                    <li>😄 <b>YOLOv8 (.pt)</b> untuk deteksi ekspresi wajah (senang, sedih, marah, takut, jijik)</li>
                    <li>🔢 <b>CNN (.h5)</b> untuk klasifikasi angka tulisan tangan 0–9</li>
                </ul>
                Setelah angka dikenali, sistem menentukan apakah angka tersebut <b>genap atau ganjil</b> secara otomatis.
                <br><br>
                Dibuat oleh <b>Ine Lutfia</b> sebagai bagian dari proyek <b>Ujian Tengah Semester Big Data & AI</b> 🎓
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.button("⬅️ Kembali ke Cover", on_click=lambda: goto("Cover"))
