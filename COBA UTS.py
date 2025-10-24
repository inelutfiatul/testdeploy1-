import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==========================
# 🌌 CONFIG
# ==========================
st.set_page_config(page_title="🚀 AI Dashboard – Ine Lutfia", page_icon="🤖", layout="wide")

# ==========================
# 🌈 CUSTOM STYLE: FUTURISTIC HOLOGRAPHIC DASHBOARD
# ==========================
st.markdown("""
<style>
/* === IMPORT FONTS === */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Poppins:wght@400;600&display=swap');

/* === BACKGROUND GRADIENT + BOKEH === */
body {
    background: radial-gradient(circle at top left, #0f0c29, #302b63, #24243e);
    color: #e0f7ff;
    font-family: 'Poppins', sans-serif;
    overflow-x: hidden;
}

/* === FLOATING BOKEH EFFECT === */
@keyframes floatBokeh {
  0% {transform: translateY(0) scale(1); opacity: 0.8;}
  100% {transform: translateY(-1200px) scale(1.5); opacity: 0;}
}
.bokeh {
  position: fixed;
  bottom: -50px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,255,255,0.8), rgba(0,255,255,0.1));
  width: 20px;
  height: 20px;
  animation: floatBokeh linear infinite;
  z-index: -1;
}

/* === TITLE === */
.title {
    font-family: 'Orbitron', sans-serif;
    text-align: center;
    font-size: 45px;
    color: #00ffff;
    text-shadow: 0 0 15px #00ffff, 0 0 40px #0077ff;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

/* === SUBHEADER === */
.subheader {
    text-align: center;
    color: #c8e6ff;
    font-size: 18px;
    margin-bottom: 40px;
}

/* === GLASS CONTAINER === */
.glass-box {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,255,255,0.15);
    backdrop-filter: blur(15px);
    transition: all 0.3s ease;
}
.glass-box:hover {
    transform: scale(1.04);
    box-shadow: 0 0 40px #00ffff;
}

/* === NEON TEXT === */
.neon-text {
    font-family: 'Orbitron', sans-serif;
    color: #00ffff;
    text-shadow: 0 0 15px #00ffff, 0 0 25px #0077ff;
    font-size: 28px;
}

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(0,0,0,0.6), rgba(20,20,50,0.85));
    border-right: 2px solid rgba(0,255,255,0.4);
    backdrop-filter: blur(15px);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] span {
    color: #aeeeff !important;
}

/* === FOOTER === */
.footer {
    text-align: center;
    color: #aeeeff;
    font-size: 13px;
    margin-top: 60px;
    opacity: 0.8;
}

/* === SCROLLBAR === */
::-webkit-scrollbar {width: 10px;}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00ffff, #0044ff);
    border-radius: 10px;
}
</style>

<!-- === BOKEH GENERATOR === -->
<script>
let count = 30;
for (let i = 0; i < count; i++) {
  let bokeh = document.createElement('div');
  bokeh.classList.add('bokeh');
  bokeh.style.left = Math.random()*100 + 'vw';
  bokeh.style.animationDuration = (5 + Math.random()*10) + 's';
  bokeh.style.width = bokeh.style.height = (5 + Math.random()*25) + 'px';
  document.body.appendChild(bokeh);
}
</script>
""", unsafe_allow_html=True)

# ==========================
# 🚀 LOAD MODELS (TIDAK DIUBAH)
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
# 🧠 HEADER
# ==========================
st.markdown("<div class='title'>🤖 AI DASHBOARD: FACE & DIGIT DETECTION</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>✨ UTS Big Data & Artificial Intelligence | by <b>Ine Lutfia</b></div>", unsafe_allow_html=True)

# ==========================
# ⚙️ SIDEBAR
# ==========================
st.sidebar.header("⚙️ Mode Analisis")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=150)
menu = st.sidebar.radio("🧩 Pilih Analisis:", ["Ekspresi Wajah", "Digit Angka"])
show_debug = st.sidebar.checkbox("Tampilkan detail prediksi", value=False)
label_offset = st.sidebar.selectbox("Offset label (jika model mulai dari 1)", [0, -1])

# ==========================
# 📤 FILE UPLOAD
# ==========================
st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# 🧩 MAIN LAYOUT (3 KOLOM GLASS CARD)
# ==========================
col1, col2, col3 = st.columns([1.2, 2, 1.2])

with col2:
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

        # 🔹 BAGIAN DETEKSI WAJAH (TIDAK DIUBAH)
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
                        <div class='glass-box'>
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

        # 🔹 BAGIAN KLASIFIKASI DIGIT (TIDAK DIUBAH)
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
                    <div class='glass-box'>
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
st.markdown("<div class='footer'>🌌 © 2025 – Ine Lutfia | AI Dashboard UTS Big Data</div>", unsafe_allow_html=True)
