import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import time
from streamlit_extras.let_it_rain import rain

# ==========================
# CONFIG & THEME
# ==========================
st.set_page_config(page_title="INÉ VISION STATION", page_icon="🪩", layout="wide")

# 🌈 CSS FUTURISTIK + ANIMASI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

html, body {
    background: radial-gradient(circle at top left, #0f172a, #1e293b);
    color: #e2e8f0;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    text-align: center;
}
.title {
    font-size: 44px;
    font-weight: 800;
    color: #38bdf8;
    text-shadow: 0 0 10px #38bdf8;
    letter-spacing: 1px;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-top: -10px;
}
.glass {
    background: rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 20px rgba(56,189,248,0.3);
    text-align: center;
    margin-top: 20px;
}
.glow {
    color: #38bdf8;
    text-shadow: 0 0 10px #38bdf8, 0 0 20px #38bdf8;
}
.footer {
    text-align:center;
    color:#94a3b8;
    margin-top:50px;
    font-size:14px;
}
.upload-box {
    border: 2px dashed #38bdf8;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    transition: 0.3s;
}
.upload-box:hover {
    border-color: #0ea5e9;
    background-color: rgba(56,189,248,0.1);
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
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

# ==========================
# HEADER
# ==========================
st.markdown("<h1 class='title'>🪩 INÉ VISION STATION</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI Dashboard for Emotion & Digit Recognition</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.image("LOGO USK.png", width=160)
st.sidebar.markdown("## 🔧 Pengaturan")
menu = st.sidebar.radio("Mode Klasifikasi", ["Ekspresi Wajah", "Digit Angka"])
show_debug = st.sidebar.checkbox("Tampilkan debug prediction", value=False)
label_offset = st.sidebar.selectbox("Label offset (kalau model 1..10)", [0, -1])

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
if uploaded_file:
    st.markdown("<div class='upload-box'>Gambar berhasil diunggah ✅</div>", unsafe_allow_html=True)

# ==========================
# MAIN SECTION
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)
    with st.spinner("⚙️ AI sedang menganalisis gambar..."):
        time.sleep(1.5)

    # ========== MODE EKSPRESI WAJAH ==========
    if menu == "Ekspresi Wajah":
        st.subheader("🧠 Deteksi Ekspresi Wajah")
        try:
            results = face_model(img)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="📸 Hasil Deteksi Wajah", use_container_width=True)

            if len(results[0].boxes) == 0:
                st.warning("⚠️ Tidak ada wajah terdeteksi. Coba unggah foto wajah lebih dekat.")
            else:
                labels = results[0].names
                emoji_map = {"senang": "😄", "bahagia": "😊", "sedih": "😢", "marah": "😡", "takut": "😱", "jijik": "🤢"}
                response_map = {
                    "senang": "Wah, kamu terlihat bahagia banget hari ini! ✨",
                    "bahagia": "Senyummu bikin dashboard ini ikut ceria 😍",
                    "sedih": "Jangan sedih, AI di sini buat nemenin kamu 💙",
                    "marah": "Coba tarik napas pelan-pelan... kamu bisa! 🌿",
                    "takut": "Tenang aja, kamu aman di sini 🔒",
                    "jijik": "Haha, ada yang bikin ilfeel ya? 😅"
                }

                for box in results[0].boxes:
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    label = labels.get(cls, "Tidak Dikenal").lower()
                    emoji = emoji_map.get(label, "🙂")
                    reaction = response_map.get(label, "Deteksi ekspresi berhasil!")

                    st.markdown(f"""
                        <div class='glass'>
                            <div class='emoji' style='font-size:60px'>{emoji}</div>
                            <h2 class='glow'>{label.capitalize()}</h2>
                            <p>🎯 Keyakinan: <b>{conf*100:.2f}%</b></p>
                            <p style='color:#e0f2fe;'>{reaction}</p>
                        </div>
                    """, unsafe_allow_html=True)

                # efek animasi pas ekspresi bahagia
                if "bahagia" in [labels[int(b.cls[0])] for b in results[0].boxes]:
                    rain(emoji="✨", font_size=20, falling_speed=3, animation_length=1)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat deteksi ekspresi: {e}")

    # ========== MODE DIGIT ANGKA ==========
    elif menu == "Digit Angka":
        st.subheader("🔢 Klasifikasi Digit Angka")
        try:
            input_shape = digit_model.input_shape
            target_size = (input_shape[1], input_shape[2])
            channels = input_shape[3] if input_shape[3] else 1

            if channels == 1:
                proc = img.convert("L")
            else:
                proc = img.convert("RGB")

            proc = proc.resize(target_size)
            arr = image.img_to_array(proc).astype("float32") / 255.0
            img_array = np.expand_dims(arr, axis=0)

            if show_debug:
                st.write("Input shape:", img_array.shape)

            pred = digit_model.predict(img_array)
            pred_label = int(np.argmax(pred[0]))
            prob = float(np.max(pred[0]))
            if label_offset == -1:
                pred_label -= 1
            pred_label %= 10

            col1, col2 = st.columns(2)
            with col1:
                st.image(proc, caption="🖼️ Gambar Preprocessed", use_column_width=True)
            with col2:
                parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"
                st.markdown(f"""
                    <div class='glass'>
                        <h2 class='glow'>Angka: {pred_label}</h2>
                        <h4>Akurasi: {prob:.2%}</h4>
                        <p>{parity}</p>
                    </div>
                """, unsafe_allow_html=True)
            if prob > 0.9:
                rain(emoji="💫", font_size=20, falling_speed=3, animation_length=1)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat klasifikasi digit: {e}")

else:
    st.markdown("<div class='upload-box'>⬆️ Silakan unggah gambar terlebih dahulu.</div>", unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown("<p class='footer'>© 2025 • INÉ VISION STATION • Proyek UTS Big Data & AI</p>", unsafe_allow_html=True)
