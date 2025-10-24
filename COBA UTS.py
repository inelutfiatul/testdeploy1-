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
st.set_page_config(page_title="AI Dashboard – Ine Lutfia", page_icon="🤖", layout="wide")

# ==========================
# 🌈 STYLING: GLOWING NEO FUTURISTIC
# ==========================
st.markdown("""
<style>
/* 🪐 BACKGROUND: Dynamic Gradient Glow */
@keyframes moveGradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
body {
    background: linear-gradient(-45deg, #020024, #090979, #00d4ff, #42006c);
    background-size: 400% 400%;
    animation: moveGradient 20s ease infinite;
    color: #E0F7FA;
    font-family: 'Poppins', sans-serif;
}

/* ✨ TITLE & SUBHEADER */
.title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: #00FFFF;
    text-shadow: 0 0 20px #00FFFF, 0 0 40px #0099FF;
    margin-bottom: 10px;
}
.subheader {
    text-align: center;
    font-size: 20px;
    color: #D9EAFD;
    margin-bottom: 40px;
    letter-spacing: 1px;
}

/* 🧊 GLASS CONTAINER */
.glass-box {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(0, 255, 255, 0.4);
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0 0 40px rgba(0,255,255,0.2);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease-in-out;
}
.glass-box:hover {
    transform: scale(1.03);
    box-shadow: 0 0 60px #00FFFF;
}

/* 🌀 NEON TEXT */
.neon-text {
    color: #00FFFF;
    font-weight: bold;
    font-size: 26px;
    text-shadow: 0 0 15px #00FFFF, 0 0 30px #00FFFF;
}

/* 🪞 SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(10, 10, 25, 0.85);
    backdrop-filter: blur(12px);
    border-right: 2px solid rgba(0,255,255,0.3);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] span {
    color: #CFFAFE !important;
}

/* 🌠 PARTICLES */
@keyframes float {
    0% { transform: translateY(0px); opacity: 0.8; }
    50% { transform: translateY(-20px); opacity: 1; }
    100% { transform: translateY(0px); opacity: 0.8; }
}
.particle {
    position: fixed;
    background: radial-gradient(circle, #00FFFF, transparent);
    border-radius: 50%;
    width: 12px;
    height: 12px;
    animation: float 4s ease-in-out infinite;
    opacity: 0.6;
    z-index: -1;
}

/* 🌙 FOOTER */
.footer {
    text-align: center;
    color: #B0E0E6;
    font-size: 14px;
    margin-top: 60px;
    text-shadow: 0 0 10px #00FFFF;
}

/* SCROLLBAR CUSTOM */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00FFFF, #004e92);
    border-radius: 10px;
}
</style>

<!-- ✨ PARTICLES GENERATOR -->
<script>
for (let i = 0; i < 30; i++) {
  let p = document.createElement('div');
  p.classList.add('particle');
  p.style.left = Math.random() * 100 + 'vw';
  p.style.top = Math.random() * 100 + 'vh';
  p.style.animationDelay = Math.random() * 5 + 's';
  p.style.width = p.style.height = (5 + Math.random() * 10) + 'px';
  document.body.appendChild(p);
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
st.markdown("<div class='title'>💎 AI Dashboard: Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>UTS Big Data & Artificial Intelligence | <b>Ine Lutfia</b></div>", unsafe_allow_html=True)

# ==========================
# ⚙️ SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=150)
menu = st.sidebar.radio("Pilih Mode:", ["Ekspresi Wajah", "Digit Angka"])
label_offset = st.sidebar.selectbox("Offset Label (Model mulai dari 1?)", [0, -1])
show_debug = st.sidebar.checkbox("Tampilkan Detail Prediksi", value=False)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# 🎭 EKSPRESI WAJAH & 🔢 ANGKA
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # 💫 DETEKSI EKSPRESI WAJAH
    if menu == "Ekspresi Wajah":
        st.subheader("🎭 Hasil Deteksi Ekspresi Wajah")
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

                emoji = {"senang": "😄", "bahagia": "😊", "sedih": "😢",
                         "marah": "😡", "takut": "😱", "jijik": "🤢"}.get(label, "🙂")

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

    # 🔢 KLASIFIKASI ANGKA
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
                st.write("📊 Prediksi Mentah:", pred)

        except Exception as e:
            st.error(f"❌ Kesalahan klasifikasi digit: {e}")

else:
    st.info("⬆️ Silakan unggah gambar untuk mulai analisis.")

# ==========================
# 🌙 FOOTER
# ==========================
st.markdown("<div class='footer'>✨ © 2025 – Ine Lutfia | UTS Big Data & AI Dashboard</div>", unsafe_allow_html=True)
