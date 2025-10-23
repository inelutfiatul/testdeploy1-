import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os, time

# ==========================
# 🌈 KONFIGURASI GLOBAL
# ==========================
st.set_page_config(page_title="AI Klasifikasi Ekspresi & Digit", page_icon="🤖", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    text-align: center; font-size: 40px; font-weight: 800; color: #A5D7E8;
    text-shadow: 0 0 15px #00FFFF;
}
.subheader { text-align: center; font-size: 18px; color: #D9EAFD; margin-top: -8px; }
.glass-box {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 0 30px rgba(0,255,255,0.15);
    text-align: center;
    backdrop-filter: blur(10px);
    transition: 0.4s;
}
.glass-box:hover { box-shadow: 0 0 35px #00FFFF; transform: scale(1.02); }
.neon-text { color: #00FFFF; text-shadow: 0 0 10px #00FFFF, 0 0 25px #00FFFF; font-weight: bold; }
.footer { text-align: center; color: #B0E0E6; font-size: 13px; margin-top: 40px; }
.btn {
    display: inline-block; background: linear-gradient(45deg, #00B4DB, #0083B0);
    color: white; padding: 12px 25px; border-radius: 30px; text-decoration: none;
    font-weight: bold; transition: 0.3s ease;
}
.btn:hover { transform: scale(1.05); background: linear-gradient(45deg, #36D1DC, #5B86E5); }
</style>
""", unsafe_allow_html=True)

# ==========================
# 🧭 NAVIGASI SLIDE (Halaman)
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

# ==========================
# 🚀 LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    face_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    digit_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    face_model = YOLO(face_path) if os.path.exists(face_path) else None
    digit_model = tf.keras.models.load_model(digit_path) if os.path.exists(digit_path) else None
    return face_model, digit_model

face_model, digit_model = load_models()

# ==========================
# 🏠 SLIDE 1 - COVER PAGE
# ==========================
if st.session_state.page == "home":
    st.markdown("<div class='title'>🤖 AI Dashboard UTS</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Klasifikasi Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("LOGO USK.png", width=180)
    st.markdown("""
        <div style='text-align:center; font-size:18px; color:#BEE9E8;'>
        Dibuat oleh <b>Ine Lutfia</b><br>
        Proyek UTS Big Data & Artificial Intelligence<br><br>
        </div>
    """, unsafe_allow_html=True)

    if st.button("✨ MULAI EKSPLORASI"):
        go_to("ekspresi")

# ==========================
# 🎭 SLIDE 2 - EKSPRESI WAJAH
# ==========================
elif st.session_state.page == "ekspresi":
    st.markdown("<div class='title'>🎭 Deteksi Ekspresi Wajah</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Unggah Gambar Wajah", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and face_model:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

        results = face_model(img)
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
            cls = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            label = results[0].names.get(cls, "Tidak Dikenal").lower()

            emoji_map = {
                "senang": "😄", "bahagia": "😊", "sedih": "😢",
                "marah": "😡", "takut": "😱", "jijik": "🤢"
            }
            emoji = emoji_map.get(label, "🙂")

            st.markdown(f"""
                <div class='glass-box'>
                    <h2 class='neon-text'>{emoji} {label.capitalize()}</h2>
                    <p>Akurasi: <b>{conf*100:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("😅 Tidak ada wajah terdeteksi.")

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Kembali ke Cover"):
            go_to("home")
    with col2:
        if st.button("➡️ Lanjut ke Klasifikasi Angka"):
            go_to("angka")

# ==========================
# 🔢 SLIDE 3 - KLASIFIKASI ANGKA
# ==========================
elif st.session_state.page == "angka":
    st.markdown("<div class='title'>🔢 Klasifikasi Digit Angka</div>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Unggah gambar angka tulisan tangan</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📥 Unggah Gambar Digit", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and digit_model:
        img = Image.open(uploaded_file).convert("L").resize((28, 28))
        arr = image.img_to_array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        with st.spinner("⏳ Menganalisis angka..."):
            time.sleep(1.2)
            pred = digit_model.predict(arr)
            label = int(np.argmax(pred[0]))
            prob = float(np.max(pred[0]))

        emoji = "✨" if prob > 0.9 else "🤔"
        warna = "#00FFFF" if prob > 0.8 else "#FFD700"

        st.markdown(f"""
            <div class='glass-box'>
                <h2 style='color:{warna}; font-size:40px;'>{emoji} Angka: {label}</h2>
                <p>Akurasi: <b>{prob*100:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

        if label % 2 == 0:
            st.success("✅ Angka ini genap! Stabil dan seimbang ⚖️")
        else:
            st.warning("⚡ Angka ganjil! Dinamis dan penuh energi 🔥")

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Kembali ke Ekspresi"):
            go_to("ekspresi")
    with col3:
        if st.button("ℹ️ Tentang Proyek"):
            go_to("tentang")

# ==========================
# 🧾 SLIDE 4 - TENTANG
# ==========================
elif st.session_state.page == "tentang":
    st.markdown("<div class='title'>📘 Tentang Proyek</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='glass-box'>
            <p>
            Dashboard ini dikembangkan untuk UTS mata kuliah <b>Big Data & Artificial Intelligence</b>.<br><br>
            🧠 Menggunakan dua model AI:<br>
            • YOLOv8 untuk deteksi ekspresi wajah<br>
            • CNN (Keras/TensorFlow) untuk klasifikasi digit angka<br><br>
            Tujuan proyek ini adalah menggabungkan <b>computer vision</b> dan <b>machine learning</b>
            dalam satu antarmuka interaktif yang modern dan informatif.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🏠 Kembali ke Cover"):
        go_to("home")

st.markdown("<div class='footer'>© 2025 – Ine Lutfia | Proyek UTS Big Data & AI ✨</div>", unsafe_allow_html=True)
