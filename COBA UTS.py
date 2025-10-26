import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==========================
# 🌌 CONFIG & STYLE - FINAL SUPER GLOWING VERSION (BLACK TITLE)
# ==========================
st.set_page_config(page_title="AI Dashboard – Ine Lutfia", page_icon="🤖", layout="wide")

# ==== STYLING ====
st.markdown("""
<style>
/* 🌌 BACKGROUND GALAXY */
@keyframes gradientMove {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
body {
  background: linear-gradient(-45deg, #000428, #004e92, #1a1a40, #001f3f);
  background-size: 400% 400%;
  animation: gradientMove 30s ease infinite;
  color: #FFFFFF;
  font-family: 'Poppins', sans-serif;
}

/* 🌟 TITLE */
.title {
  text-align: center;
  font-size: 46px;
  font-weight: 800;
  color: #000000; /* 🖤 WARNA JUDUL HITAM */
  text-shadow: 0 0 25px #00FFFF, 0 0 40px #00FFFF, 0 0 60px #00FFFF;
  margin-bottom: 10px;
  letter-spacing: 2px;
}

/* 💫 SUBHEADER */
.subheader {
  text-align: center;
  font-size: 18px;
  color: #000000; /* 🖤 WARNA SUBJUDUL HITAM */
  margin-bottom: 45px;
  text-shadow: 0 0 12px rgba(0,255,255,0.6);
}

/* 🧊 GLASS CONTAINER */
.glass-box {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.25);
  border-radius: 18px;
  padding: 25px;
  margin-top: 15px;
  box-shadow: 0 0 30px rgba(0, 255, 255, 0.15);
  backdrop-filter: blur(12px);
  text-align: center;
  transition: 0.4s;
}
.glass-box:hover {
  transform: scale(1.03);
  box-shadow: 0 0 45px rgba(0, 255, 255, 0.5);
}

/* ✨ NEON TEXT */
.neon-text {
  color: #FFFFFF;
  text-shadow: 
    0 0 10px #00FFFF,
    0 0 20px #00FFFF,
    0 0 40px #00FFFF;
  font-weight: 700;
  font-size: 26px;
}

/* ⚙️ SIDEBAR */
[data-testid="stSidebar"] {
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(12px);
  border-right: 2px solid rgba(0,255,255,0.3);
}
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label {
  color: #E0FFFF !important;
  text-shadow: 0 0 8px rgba(0,255,255,0.4);
}

/* 🧿 FOOTER */
.footer {
  text-align: center;
  color: #B0E0E6;
  font-size: 13px;
  margin-top: 60px;
  opacity: 0.8;
  text-shadow: 0 0 10px rgba(0,255,255,0.5);
}

/* 🌠 IMAGE BORDER EFFECT */
img {
  border-radius: 15px;
  box-shadow: 0 0 25px rgba(0,255,255,0.3);
}

/* 🌌 BUTTON STYLE */
.stButton button {
  background: linear-gradient(90deg, #00FFFF, #0072ff);
  color: #000;
  font-weight: 700;
  border-radius: 12px;
  border: none;
  padding: 10px 24px;
  transition: 0.3s;
}
.stButton button:hover {
  background: linear-gradient(90deg, #0072ff, #00FFFF);
  box-shadow: 0 0 20px #00FFFF;
  color: #fff;
}

/* ✨ SCROLLBAR */
::-webkit-scrollbar {
  width: 10px;
}
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #00FFFF, #004e92);
  border-radius: 10px;
}

/* 💫 FLOATING PARTICLES */
@keyframes floatUp {
  0% { transform: translateY(0) scale(1); opacity: 0.8; }
  100% { transform: translateY(-800px) scale(0.3); opacity: 0; }
}
.particle {
  position: fixed;
  bottom: -50px;
  background: rgba(0, 255, 255, 0.7);
  border-radius: 50%;
  width: 8px;
  height: 8px;
  animation: floatUp linear infinite;
  z-index: -1;
}

/* 💬 INFO BOX */
.stAlert {
  background: rgba(255,255,255,0.1);
  border-left: 4px solid #00FFFF !important;
  color: #E0FFFF !important;
}
</style>

<!-- 🌌 PARTICLE GENERATOR -->
<script>
let particles = 25;
for (let i = 0; i < particles; i++) {
  let p = document.createElement('div');
  p.classList.add('particle');
  p.style.left = Math.random() * 100 + 'vw';
  p.style.animationDuration = (6 + Math.random() * 6) + 's';
  p.style.width = p.style.height = (4 + Math.random() * 12) + 'px';
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
st.markdown("<div class='title'>🌌 AI Dashboard: Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>UTS Big Data & Artificial Intelligence | by <b>Ine Lutfia</b></div>", unsafe_allow_html=True)

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

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    with col2:
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
