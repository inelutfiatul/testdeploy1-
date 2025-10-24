import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==========================
# 🌌 CONFIG & STYLE
# ==========================
st.set_page_config(page_title="AI Dashboard – Ine Lutfia", page_icon="🤖", layout="wide")

# ==== ULTRA STYLISH CUSTOM THEME ====
st.markdown("""
<style>

/* ========================== */
/* 🌠 BACKGROUND AURORA GLOW */
/* ========================== */
html, body, [class*="css"] {
    background: radial-gradient(circle at top left, #010018, #001E44 50%, #000000);
    color: #E0F7FA;
    font-family: 'Poppins', sans-serif;
    overflow-x: hidden;
}

@keyframes aurora {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
body {
    background: linear-gradient(-45deg, #000428, #004e92, #00182f, #0a1a33);
    background-size: 400% 400%;
    animation: aurora 30s ease infinite;
}

/* ========================== */
/* ✨ TITLES & HEADINGS */
/* ========================== */
.title {
    text-align: center;
    font-size: 46px;
    font-weight: 900;
    color: #00FFFF;
    text-shadow: 0 0 10px #00FFFF, 0 0 25px #00FFFF;
    margin-bottom: 10px;
    animation: pulse 3s infinite alternate;
}
@keyframes pulse {
    0% {text-shadow: 0 0 10px #00FFFF;}
    100% {text-shadow: 0 0 35px #00FFFF;}
}

.subheader {
    text-align: center;
    font-size: 18px;
    color: #A6E8F5;
    margin-bottom: 40px;
}

/* ========================== */
/* 🧊 GLASS PANEL DESIGN */
/* ========================== */
.glass-panel {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    border: 1px solid rgba(0, 255, 255, 0.2);
    box-shadow: 0 0 40px rgba(0,255,255,0.1);
    padding: 25px;
    backdrop-filter: blur(15px);
    transition: 0.3s;
}
.glass-panel:hover {
    transform: scale(1.02);
    box-shadow: 0 0 50px rgba(0,255,255,0.4);
}

/* ========================== */
/* 🌌 SIDEBAR REWORK */
/* ========================== */
[data-testid="stSidebar"] {
    background: rgba(0, 10, 30, 0.8);
    border-right: 2px solid rgba(0,255,255,0.3);
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] span {
    color: #B0E0E6 !important;
}

/* ========================== */
/* 🌟 CUSTOM BUTTONS */
/* ========================== */
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 0 10px rgba(0,255,255,0.4);
    transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #00FFFF, #004e92);
    box-shadow: 0 0 25px #00FFFF;
}

/* ========================== */
/* 🪞 SCROLLBAR */
/* ========================== */
::-webkit-scrollbar {width: 10px;}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00FFFF, #004e92);
    border-radius: 10px;
}

/* ========================== */
/* 🌙 FOOTER */
/* ========================== */
.footer {
    text-align: center;
    font-size: 13px;
    color: #A8EFFF;
    margin-top: 60px;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# 🚀 LOAD MODELS
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
# 💫 HEADER
# ==========================
st.markdown("<div class='title'>🌌 AI Galaxy Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>UTS Big Data & Artificial Intelligence • <b>Ine Lutfia</b></div>", unsafe_allow_html=True)

# ==========================
# ⚙️ SIDEBAR
# ==========================
st.sidebar.header("⚙️ Navigasi")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=140)
menu = st.sidebar.radio("🔍 Pilih Mode Analisis:", ["Ekspresi Wajah", "Digit Angka"])
show_debug = st.sidebar.checkbox("Tampilkan detail prediksi", value=False)
label_offset = st.sidebar.selectbox("Offset label (jika model mulai dari 1)", [0, -1])

# ==========================
# 📤 FILE UPLOAD
# ==========================
st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Unggah Gambar untuk Analisis", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# 🧠 MAIN SECTION
# ==========================
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    with col2:
        # === LOGIKA DETEKSI (TIDAK DIUBAH) ===
        if menu == "Ekspresi Wajah":
            st.subheader("🎭 Deteksi Ekspresi Wajah")
            try:
                results = face_model(img)
                annotated_img = results[0].plot()
                st.image(annotated_img, caption="📸 Deteksi Wajah", use_container_width=True)

                if len(results[0].boxes) == 0:
                    st.warning("😅 Tidak ada wajah terdeteksi.")
                else:
                    boxes = results[0].boxes
                    best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
                    cls = int(best_box.cls[0])
                    conf = float(best_box.conf[0])
                    label = results[0].names.get(cls, "Tidak Dikenal").lower()

                    emoji = {
                        "senang": "😄", "bahagia": "😊", "sedih": "😢",
                        "marah": "😡", "takut": "😱", "jijik": "🤢"
                    }.get(label, "🙂")

                    st.markdown(f"""
                        <div class='glass-panel'>
                            <h2 class='neon-text'>{emoji} {label.capitalize()}</h2>
                            <p>Akurasi Deteksi: <b>{conf*100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                    if label in ["sedih", "takut"]:
                        st.info("💬 Jangan khawatir, semuanya akan baik-baik aja 🌈")
                    elif label in ["bahagia", "senang"]:
                        st.success("💬 Senyummu bikin dunia lebih cerah hari ini! 😄☀️")
                    elif label == "marah":
                        st.warning("💬 Yuk tarik napas dulu, kamu pasti bisa kendalikan emosi 💪")

            except Exception as e:
                st.error(f"❌ Kesalahan deteksi wajah: {e}")

        elif menu == "Digit Angka":
            st.subheader("🔢 Klasifikasi Angka Tulisan Tangan")
            try:
                input_shape = digit_model.input_shape
                size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 else (28, 28)
                channels = input_shape[3] if len(input_shape) == 4 else 1

                proc = img.convert("L" if channels == 1 else "RGB").resize(size)
                arr = image.img_to_array(proc).astype("float32") / 255.0
                arr = np.expand_dims(arr, axis=0)

                pred = digit_model.predict(arr)
                pred_label = int(np.argmax(pred[0]))
                prob = float(np.max(pred[0]))
                if label_offset == -1:
                    pred_label -= 1
                pred_label = pred_label % 10
                parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"

                st.markdown(f"""
                    <div class='glass-panel'>
                        <h2 class='neon-text'>Angka: {pred_label}</h2>
                        <p>Akurasi: <b>{prob*100:.2f}%</b></p>
                        <p>{parity}</p>
                    </div>
                """, unsafe_allow_html=True)

                if show_debug:
                    st.write("📊 Detail Prediksi Mentah:", pred)

            except Exception as e:
                st.error(f"❌ Kesalahan klasifikasi digit: {e}")

else:
    st.info("⬆️ Silakan unggah gambar untuk mulai analisis.")

# ==========================
# 🌙 FOOTER
# ==========================
st.markdown("<div class='footer'>✨ © 2025 – Ine Lutfia | UTS Big Data & AI Dashboard</div>", unsafe_allow_html=True)
