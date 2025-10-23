import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import time

# ==========================
# CONFIG & STYLING
# ==========================
st.set_page_config(page_title="🧠 AI Vision Dashboard", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #E0EAFC 0%, #CFDEF3 100%);
        font-family: 'Poppins', sans-serif;
    }
    .title {
        text-align: center; 
        font-size: 36px; 
        color: #4B7BE5; 
        font-weight: bold; 
        margin-bottom: 0;
    }
    .subheader {
        text-align: center; 
        font-size: 20px; 
        color: #333; 
        margin-top: -5px;
        margin-bottom: 20px;
    }
    .glass-box {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        text-align: center;
        margin-top: 15px;
    }
    .emoji {
        font-size: 60px;
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
st.markdown("<div class='title'>🤖 AI Vision Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Klasifikasi Ekspresi Wajah & Digit Angka | Proyek UTS Big Data & AI</div>", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan")
logo_path = "LOGO USK.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=140)
else:
    st.sidebar.warning("⚠️ Logo tidak ditemukan")

menu = st.sidebar.radio("Pilih Jenis Klasifikasi:", ["Ekspresi Wajah", "Digit Angka"])
label_offset = st.sidebar.selectbox("Label offset (jika model 1..10)", options=[0, -1])
show_debug = st.sidebar.checkbox("Tampilkan debug prediction vector", value=False)
dark_mode = st.sidebar.toggle("🌙 Mode Gelap", value=False)

if dark_mode:
    st.markdown("<style>body{background-color:#1E1E1E;color:white;}</style>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# MAIN LOGIC
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # progress loading
    with st.spinner("⏳ AI sedang menganalisis gambar..."):
        time.sleep(1.5)

    # -----------------------------------------
    # MODE 1: EKSPRESI WAJAH
    # -----------------------------------------
    if menu == "Ekspresi Wajah":
        st.subheader("🔍 Hasil Deteksi Ekspresi Wajah")
        try:
            results = face_model(img)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="📸 Deteksi Ekspresi", use_container_width=True)

            if len(results[0].boxes) == 0:
                st.warning("⚠️ Tidak ada wajah terdeteksi. Coba unggah gambar wajah lebih dekat.")
            else:
                model_labels = results[0].names
                emoji_map = {
                    "senang": "😄", "bahagia": "😊", "sedih": "😢",
                    "marah": "😡", "takut": "😱", "jijik": "🤢",
                }
                personality_responses = {
                    "senang": "Wah, kamu terlihat bahagia hari ini! 😄",
                    "bahagia": "Senangnya lihat kamu tersenyum! 😊",
                    "sedih": "Semangat ya, semua akan baik-baik saja 💪",
                    "marah": "Tenangkan diri dulu, ambil napas dalam-dalam 😌",
                    "takut": "Jangan khawatir, kamu aman di sini 🫶",
                    "jijik": "Ups... ada yang bikin risih ya? 😅",
                }

                for box in results[0].boxes:
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    label = model_labels.get(cls, "Tidak Dikenal").lower()
                    emoji = emoji_map.get(label, "🙂")
                    st.markdown(f"""
                        <div class='glass-box'>
                            <div class='emoji'>{emoji}</div>
                            <h3>Ekspresi: <b>{label.capitalize()}</b></h3>
                            <p>🎯 Keyakinan: <b>{conf*100:.2f}%</b></p>
                            <p>{personality_responses.get(label, '')}</p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat deteksi ekspresi: {e}")

    # -----------------------------------------
    # MODE 2: DIGIT ANGKA
    # -----------------------------------------
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Digit Angka")
        try:
            input_shape = digit_model.input_shape
            if len(input_shape) != 4:
                target_size = (28, 28)
                channels = 1
            else:
                target_size = (input_shape[1], input_shape[2])
                channels = input_shape[3] if input_shape[3] is not None else 1

            if channels == 1:
                proc = img.convert("L")
            else:
                proc = img.convert("RGB")

            proc = proc.resize(target_size)
            arr = image.img_to_array(proc).astype("float32") / 255.0
            if arr.ndim == 3 and arr.shape[2] != channels:
                if channels == 1:
                    proc = proc.convert("L")
                    arr = image.img_to_array(proc).astype("float32") / 255.0
                else:
                    proc = proc.convert("RGB")
                    arr = image.img_to_array(proc).astype("float32") / 255.0

            img_array = np.expand_dims(arr, axis=0)

            if show_debug:
                st.info(f"Model input shape: {input_shape}")
                with st.expander("Debug Info"):
                    st.write("Input array shape:", img_array.shape)
                    st.write("Min/Max:", float(img_array.min()), float(img_array.max()))

            pred = digit_model.predict(img_array)
            pred_label = int(np.argmax(pred[0]))
            prob = float(np.max(pred[0]))
            if label_offset == -1:
                pred_label = pred_label - 1
            pred_label = int(pred_label) % 10

            col1, col2 = st.columns(2)
            with col1:
                st.image(proc, caption="🖼️ Gambar Uji (Preprocessed)", use_column_width=True)
            with col2:
                parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"
                st.markdown(f"""
                    <div class='glass-box'>
                        <h2>Angka Terdeteksi: <b>{pred_label}</b></h2>
                        <h4>Akurasi: {prob:.2%}</h4>
                        <p>{parity}</p>
                    </div>
                """, unsafe_allow_html=True)

            if prob > 0.9:
                st.balloons()

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat klasifikasi digit: {e}")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk melakukan deteksi atau klasifikasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 – Dibuat oleh <b>Ine Lutfia</b> • Proyek UTS Big Data & AI</p>", unsafe_allow_html=True)
