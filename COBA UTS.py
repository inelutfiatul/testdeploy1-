import streamlit as st
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model

# ==============================
# CONFIGURASI AWAL
# ==============================
st.set_page_config(page_title="AI Dashboard UTS", page_icon="🤖", layout="wide")

# ===== CSS GLOW & NAVIGATION STYLE =====
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
</style>
""", unsafe_allow_html=True)

# ==============================
# SIMULASI MODEL
# ==============================
# Ganti dengan model asli kamu
# digit_model = load_model('model_digit.h5')
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Untuk testing tanpa model:
class DummyModel:
    def predict(self, x):
        return np.random.rand(1,10)
digit_model = DummyModel()

# ==============================
# SISTEM NAVIGASI SLIDE
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
    if st.button("👁️ Deteksi Wajah", key="face"): goto("Face Detection")
with col3:
    if st.button("🔢 Klasifikasi Angka", key="digit"): goto("Digit Classifier")
with col4:
    if st.button("💡 Tentang AI-ku", key="about"): goto("About")
st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# HALAMAN 1 — COVER
# ==============================
if st.session_state.page == "Cover":
    st.markdown("""
        <div class="neon-box">
            <h1>🤖 DASHBOARD AI UTS</h1>
            <h3>Oleh: <span style='color:#fff'>Ine Lutfiatul Hanifah</span></h3>
            <p style='text-align:center; color:#ccc;'>
                Selamat datang di Dashboard AI Futuristik yang dibuat khusus untuk Ujian Tengah Semester dan Laporan.<br>
                Jelajahi kemampuan AI dalam mendeteksi wajah & mengenali angka secara real-time!
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("➡️ Mulai Jelajah", on_click=lambda: goto("Face Detection"))

# ==============================
# HALAMAN 2 — DETEKSI WAJAH
# ==============================
elif st.session_state.page == "Face Detection":
    st.markdown("<h2>👁️ Deteksi Wajah</h2>", unsafe_allow_html=True)
    uploaded_face = st.file_uploader("Upload gambar wajah", type=["jpg","jpeg","png"])

    if uploaded_face is not None:
        img = Image.open(uploaded_face)
        st.image(img, caption="Gambar Diupload", width=300)
        st.success("✅ Wajah berhasil terdeteksi! (simulasi)")
    else:
        st.info("Unggah foto wajah untuk mendeteksi ekspresi 😊")

    st.button("⬅️ Kembali ke Cover", on_click=lambda: goto("Cover"))
    st.button("➡️ Lanjut ke Klasifikasi Angka", on_click=lambda: goto("Digit Classifier"))

# ==============================
# HALAMAN 3 — KLASIFIKASI ANGKA
# ==============================
elif st.session_state.page == "Digit Classifier":
    st.markdown("<h2>🔢 Klasifikasi Angka</h2>", unsafe_allow_html=True)
    uploaded_digit = st.file_uploader("Upload gambar angka tulisan tangan", type=["jpg","jpeg","png"])

    if uploaded_digit is not None:
        img = Image.open(uploaded_digit).convert('L')
        img = img.resize((28,28))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))
        pred = digit_model.predict(arr)
        digit = np.argmax(pred)
        st.image(img, caption="Gambar diproses", width=150)
        st.success(f"✨ AI menebak angka ini adalah: **{digit}** (akurasi {np.max(pred)*100:.2f}%)")
    else:
        st.info("Unggah gambar angka untuk diklasifikasi 🔍")

    st.button("⬅️ Kembali ke Deteksi Wajah", on_click=lambda: goto("Face Detection"))
    st.button("➡️ Tentang AI-ku", on_click=lambda: goto("About"))

# ==============================
# HALAMAN 4 — TENTANG AI-KU
# ==============================
elif st.session_state.page == "About":
    st.markdown("""
        <div class="neon-box">
            <h2>💡 Tentang AI-ku</h2>
            <p style='text-align:center; color:#ccc;'>
                Dashboard ini dibangun menggunakan <b>Streamlit</b> dan <b>Keras</b>.<br>
                Fitur utama:
                <ul>
                    <li>Deteksi wajah dengan OpenCV</li>
                    <li>Klasifikasi angka dengan model CNN (Convolutional Neural Network)</li>
                    <li>Tampilan futuristik dengan tema neon & navigasi interaktif</li>
                </ul>
                Proyek ini dibuat sebagai bagian dari <b>Ujian Tengah Semester</b>.<br>
                Desain dan animasi dibuat untuk menciptakan pengalaman interaktif seperti AI LAB masa depan. 🚀
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.button("⬅️ Kembali ke Cover", on_click=lambda: goto("Cover"))
