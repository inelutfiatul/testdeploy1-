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

# ==== CUSTOM STYLING ====
st.markdown("""
<style>
/* 🌌 BACKGROUND GALAXY ANIMATED */
@keyframes moveBackground {
    from {background-position: 0 0;}
    to {background-position: 1000px 1000px;}
}
body {
    background: radial-gradient(circle at top, #000428, #004e92);
    background-size: 400% 400%;
    animation: moveBackground 60s linear infinite;
    color: white;
    font-family: 'Poppins', sans-serif;
}

/* ✨ NEON TITLE */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #00FFFF;
    text-shadow: 0 0 20px #00FFFF, 0 0 40px #00FFFF;
    margin-bottom: 10px;
}

/* 💫 SUBHEADER */
.subheader {
    text-align: center;
    font-size: 18px;
    color: #D0F0FF;
    margin-bottom: 40px;
}

/* 🧊 GLASS CONTAINER */
.glass-box {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 4px 30px rgba(0, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    text-align: center;
    transition: all 0.3s ease-in-out;
}
.glass-box:hover {
    transform: scale(1.03);
    box-shadow: 0 0 35px #00FFFF;
}

/* 🌟 FLOATING PARTICLES */
@keyframes floatParticle {
    0% { transform: translateY(0) rotate(0deg); opacity: 1; }
    100% { transform: translateY(-800px) rotate(720deg); opacity: 0; }
}
.particle {
    position: fixed;
    bottom: -50px;
    background: rgba(0, 255, 255, 0.8);
    border-radius: 50%;
    width: 10px;
    height: 10px;
    animation: floatParticle linear infinite;
    z-index: -1;
}

/* 🎯 NEON TEXT */
.neon-text {
    color: #00FFFF;
    text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF;
    font-weight: bold;
    font-size: 24px;
}

/* 🪞 SIDEBAR STYLE */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(12px);
    border-right: 2px solid rgba(0,255,255,0.3);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] span {
    color: #B0E0E6 !important;
}

/* 🧿 FOOTER */
.footer {
    text-align: center;
    color: #B0E0E6;
    font-size: 13px;
    margin-top: 60px;
    opacity: 0.8;
}

/* 🌠 SCROLLBAR CUSTOM */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00FFFF, #004e92);
    border-radius: 10px;
}
</style>

<!-- 🔮 PARTICLE GENERATOR -->
<script>
let particleCount = 25;
for (let i = 0; i < particleCount; i++) {
    let particle = document.createElement('div');
    particle.classList.add('particle');
    particle.style.left = Math.random() * 100 + 'vw';
    particle.style.animationDuration = (5 + Math.random() * 5) + 's';
    particle.style.width = particle.style.height = (5 + Math.random() * 10) + 'px';
    document.body.appendChild(particle);
}
</script>
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
# 🧠 HEADER
# ==========================
st.markdown("<div class='title'>🌌 AI Dashboard: Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Proyek UTS – Big Data & Artificial Intelligence | by <b>Ine Lutfia</b></div>", unsafe_allow_html=True)

# ==========================
# ⚙️ SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan Mode")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=150)
menu = st.sidebar.radio("Pilih Analisis:", ["Ekspresi Wajah", "Digit Angka"])
show_debug = st.sidebar.checkbox("Tampilkan detail prediksi", value=False)
label_offset = st.sidebar.selectbox("Offset label (jika model mulai dari 1)", [0, -1])

# ==========================
# 📤 FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# 🧩 MAIN LOGIC
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # 1️⃣ EKSPRESI WAJAH
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

                # Respons interaktif
                if label in ["sedih", "takut"]:
                    st.info("💬 Jangan khawatir, semuanya akan baik-baik aja 🌈")
                elif label in ["bahagia", "senang"]:
                    st.success("💬 Senyummu bikin dunia lebih cerah hari ini! 😄☀️")
                elif label == "marah":
                    st.warning("💬 Yuk tarik napas dulu, kamu pasti bisa kendalikan emosi 💪")

        except Exception as e:
            st.error(f"❌ Kesalahan deteksi wajah: {e}")

    # 2️⃣ DIGIT ANGKA
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
st.markdown("<div class='footer'>✨ © 2025 – Ine Lutfia | UTS Big Data & AI Dashboard</div>", unsafe_allow_html=True)
