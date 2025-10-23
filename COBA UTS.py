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
st.set_page_config(page_title="🤖 AI Vision UTS", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at top, #0F2027, #203A43, #2C5364);
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        color: #00FFFF;
        text-shadow: 0 0 20px #00FFFF;
        letter-spacing: 2px;
    }
    .subtitle {
        text-align: center;
        color: #B8E4F0;
        font-size: 18px;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .glass-box {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 25px;
        text-align: center;
        transition: 0.3s ease-in-out;
    }
    .glass-box:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.5);
    }
    .neon-text {
        font-size: 24px;
        color: #7DF9FF;
        text-shadow: 0px 0px 12px #00FFFF;
    }
    .ai-message {
        background: rgba(0, 255, 255, 0.05);
        border-left: 5px solid #00FFFF;
        padding: 10px 20px;
        margin: 15px 0;
        font-style: italic;
        border-radius: 10px;
        animation: fadeIn 1.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .emoji-burst {
        text-align: center;
        font-size: 36px;
        animation: float 1.2s infinite alternate;
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
# 💬 HEADER
# ==========================
st.markdown("<div class='title'>AI Vision Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>UTS – Big Data & Artificial Intelligence by Ine Lutfia</div>", unsafe_allow_html=True)

# ==========================
# 🎛️ SIDEBAR
# ==========================
st.sidebar.title("⚙️ Pengaturan")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=130)
menu = st.sidebar.radio("Pilih Mode:", ["Ekspresi Wajah 🤖", "Digit Angka 🔢"])
label_offset = st.sidebar.selectbox("Label Offset", [0, -1])
st.sidebar.markdown("---")
st.sidebar.caption("💡 Dibuat dengan cinta oleh Ine Lutfia")

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# ✨ ANIMASI VISUAL
# ==========================
def ai_intro():
    st.markdown("<div class='emoji-burst'>🌌 🤖 💫</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Sistem AI sedang mempersiapkan analisis...</p>", unsafe_allow_html=True)
    time.sleep(1.2)

def ai_message(msg):
    st.markdown(f"<div class='ai-message'>🤖 <b>AI:</b> {msg}</div>", unsafe_allow_html=True)

# ==========================
# 🔍 LOGIKA KLASIFIKASI
# ==========================
if uploaded_file is not None:
    ai_intro()
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if menu == "Ekspresi Wajah 🤖":
        st.subheader("🎭 Deteksi Ekspresi Wajah")
        try:
            results = face_model(img)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="📸 Deteksi Ekspresi", use_container_width=True)

            if len(results[0].boxes) == 0:
                ai_message("Saya tidak mendeteksi wajah yang jelas. Coba unggah foto close-up ya! 😅")
            else:
                model_labels = results[0].names
                for box in results[0].boxes:
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    label = model_labels.get(cls, "Tidak Dikenal").lower()

                    emoji = {
                        "senang": "😄", "bahagia": "😊", "sedih": "😢",
                        "marah": "😡", "takut": "😱", "jijik": "🤢",
                    }.get(label, "🙂")

                    ai_message(f"Saya mendeteksi ekspresi <b>{label}</b> dengan keyakinan {conf*100:.2f}%. {emoji}")
                    if label in ["sedih", "takut"]:
                        ai_message("Hei, kamu tidak sendiri! 😊 Semangat ya 💪")
                    elif label in ["bahagia", "senang"]:
                        ai_message("Wah, aku suka lihat senyummu! 😄✨")
                    elif label == "marah":
                        ai_message("Tenang dulu ya... napas dalam dulu 😤🫶")

                    st.markdown(f"""
                        <div class='glass-box'>
                            <p class='neon-text'>{emoji} {label.capitalize()}</p>
                            <p>Akurasi: <b>{conf*100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            ai_message(f"Terjadi kesalahan: {e}")

    elif menu == "Digit Angka 🔢":
        st.subheader("🔢 Klasifikasi Digit Angka")
        try:
            input_shape = digit_model.input_shape
            target_size = (input_shape[1], input_shape[2])
            channels = input_shape[3] if len(input_shape) == 4 else 1

            proc = img.convert("L" if channels == 1 else "RGB").resize(target_size)
            arr = image.img_to_array(proc).astype("float32") / 255.0
            img_array = np.expand_dims(arr, axis=0)

            pred = digit_model.predict(img_array)
            pred_label = int(np.argmax(pred[0]))
            prob = float(np.max(pred[0]))
            if label_offset == -1:
                pred_label -= 1
            pred_label = pred_label % 10

            ai_message(f"Saya memprediksi angka ini adalah <b>{pred_label}</b> dengan keyakinan {prob:.2%}.")
            parity = "GENAP ✅" if pred_label % 2 == 0 else "GANJIL ⚠️"
            st.markdown(f"""
                <div class='glass-box'>
                    <h1 style='color:#00FFFF; text-shadow:0 0 15px #00FFFF;'>{pred_label}</h1>
                    <p class='neon-text'>{parity}</p>
                    <p>Akurasi: {prob:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

            if prob > 0.9:
                ai_message("Aku sangat yakin dengan prediksi ini! 🚀")
            else:
                ai_message("Hmm... sepertinya aku masih ragu. Coba gambar lain ya 🤔")

        except Exception as e:
            ai_message(f"⚠️ Terjadi kesalahan saat klasifikasi: {e}")

else:
    ai_message("Silakan unggah gambar agar saya bisa mulai analisisnya ✨")

# ==========================
# 🧩 FOOTER
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#7DF9FF;'>© 2025 – Ine Lutfia • UTS Big Data & AI</p>", unsafe_allow_html=True)
