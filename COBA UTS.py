import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import random
import time

# ==========================
# ⚙️ CONFIG & STYLE
# ==========================
st.set_page_config(page_title="AI Dashboard UTS – Ekspresi & Digit", page_icon="🤖", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .title {
        text-align: center; 
        font-size: 36px; 
        color: #A5C9FF; 
        font-weight: 700;
        text-shadow: 0px 0px 15px #1E90FF;
    }
    .subheader {
        color: #DDD; 
        font-size: 20px; 
        text-align: center; 
        margin-top: -10px;
        letter-spacing: 0.5px;
    }
    .result-box {
        background: rgba(255,255,255,0.08);
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        padding: 25px;
        text-align: center;
        margin-top: 20px;
        backdrop-filter: blur(8px);
        transition: 0.4s;
    }
    .result-box:hover {
        transform: scale(1.03);
        box-shadow: 0 0 20px #00B4FF;
    }
    .emoji-rain {
        text-align: center;
        font-size: 32px;
        animation: float 1.5s infinite alternate;
    }
    @keyframes float {
        from { transform: translateY(0px); }
        to { transform: translateY(10px); }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 🧠 LOAD MODELS
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
# 🎨 HEADER
# ==========================
st.markdown("<div class='title'>🤖 Dashboard AI – Deteksi Ekspresi & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Proyek UTS – Big Data & Artificial Intelligence</div>", unsafe_allow_html=True)
st.write("")

# ==========================
# 🎛️ SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan Dashboard")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=140)
menu = st.sidebar.radio("Pilih Jenis Klasifikasi:", ["Ekspresi Wajah", "Digit Angka"])
label_offset = st.sidebar.selectbox("Label offset (kalau model melabeli 1..10)", options=[0, -1])
show_debug = st.sidebar.checkbox("Tampilkan Debug Info", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("💡 <i>Dikembangkan oleh Ine Lutfia</i>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# 🔮 ANIMASI EMOJI (GANTI 'RAIN')
# ==========================
def emoji_animation():
    emojis = ["✨", "💫", "🌟", "🔮", "💎", "🚀", "🤖"]
    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"<div class='emoji-rain'>{random.choice(emojis)}</div>", unsafe_allow_html=True)

# ==========================
# 🔍 LOGIKA KLASIFIKASI
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)
    emoji_animation()  # tampilkan animasi atas

    # =====================
    # MODE 1 – EKSPRESI WAJAH
    # =====================
    if menu == "Ekspresi Wajah":
        st.subheader("🎭 Hasil Deteksi Ekspresi Wajah")
        try:
            results = face_model(img)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="📸 Deteksi Ekspresi", use_container_width=True)

            if len(results[0].boxes) == 0:
                st.warning("⚠️ Tidak ada wajah terdeteksi. Coba unggah foto wajah lebih jelas.")
            else:
                model_labels = results[0].names
                emoji_map = {
                    "senang": "😄", "bahagia": "😊", "sedih": "😢",
                    "marah": "😡", "takut": "😱", "jijik": "🤢",
                }

                for box in results[0].boxes:
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    label = model_labels.get(cls, "Tidak Dikenal").lower()
                    emoji = emoji_map.get(label, "🙂")

                    # efek "AI personality"
                    if label == "sedih":
                        msg = "Jangan sedih ya! 😊"
                    elif label == "bahagia" or label == "senang":
                        msg = "Kamu kelihatan bahagia banget hari ini! 😄"
                    elif label == "marah":
                        msg = "Wah... tenang dulu ya 😤"
                    else:
                        msg = "Ekspresimu menarik banget!"

                    st.markdown(f"""
                        <div class='result-box'>
                            <h2>{emoji} Ekspresi: <b>{label.capitalize()}</b></h2>
                            <p>🎯 Keyakinan: <b>{conf*100:.2f}%</b></p>
                            <p>{msg}</p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat deteksi ekspresi: {e}")

    # =====================
    # MODE 2 – DIGIT ANGKA
    # =====================
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Digit Angka")
        try:
            input_shape = digit_model.input_shape
            target_size = (input_shape[1], input_shape[2])
            channels = input_shape[3] if len(input_shape) == 4 else 1

            if channels == 1:
                proc = img.convert("L")
            else:
                proc = img.convert("RGB")

            proc = proc.resize(target_size)
            arr = image.img_to_array(proc).astype("float32") / 255.0
            img_array = np.expand_dims(arr, axis=0)

            pred = digit_model.predict(img_array)
            pred_label = int(np.argmax(pred[0]))
            prob = float(np.max(pred[0]))
            if label_offset == -1:
                pred_label = pred_label - 1
            pred_label = int(pred_label) % 10

            col1, col2 = st.columns(2)
            with col1:
                st.image(proc, caption="🖼️ Gambar Preprocessed", use_container_width=True)
            with col2:
                parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"
                st.markdown(f"""
                    <div class='result-box'>
                        <h1>{pred_label}</h1>
                        <p style='font-size:18px;'>Akurasi: {prob:.2%}</p>
                        <p>{parity}</p>
                    </div>
                """, unsafe_allow_html=True)
                if prob > 0.9:
                    st.success("🚀 Model sangat yakin dengan prediksi ini!")
                else:
                    st.info("🤔 Keyakinan model sedang, coba gambar lain.")

        except Exception as e:
            st.error("❌ Terjadi kesalahan saat klasifikasi digit:")
            st.error(e)

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk memulai deteksi.")
    emoji_animation()

# ==========================
# ⚡ FOOTER
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#A5C9FF;'>© 2025 • Ine Lutfia • UTS Big Data & AI</p>", unsafe_allow_html=True)
