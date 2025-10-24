import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2, os, base64

# =====================================
# 🌌 CONFIG & ANIMATED BACKGROUND
# =====================================
st.set_page_config(page_title="🤖 UTS Ine Lutfia | Big Data & AI", layout="wide")

def set_bg_animated():
    particle_bg = """
    <style>
    body {
        background-color: #0b132b;
        color: #e0f7fa;
        font-family: 'Poppins', sans-serif;
        overflow-x: hidden;
    }
    #particles-js {
        position: fixed;
        width: 100%;
        height: 100%;
        z-index: -1;
        top: 0;
        left: 0;
    }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
    <script>
    particlesJS('particles-js', {
      "particles": {
        "number": {"value": 120},
        "size": {"value": 3},
        "move": {"speed": 1},
        "color": {"value": "#00ffff"},
        "line_linked": {"enable": true, "color": "#00ffff"}
      }
    });
    </script>
    <div id="particles-js"></div>
    """
    st.markdown(particle_bg, unsafe_allow_html=True)

set_bg_animated()

# =====================================
# 💾 LOAD MODELS
# =====================================
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

    return YOLO(face_path), tf.keras.models.load_model(digit_path)

face_model, digit_model = load_models()

# =====================================
# 🧭 NAVIGASI MULTI-HALAMAN
# =====================================
st.sidebar.title("🚀 Navigasi Dashboard")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Cover", "🎭 Ekspresi Wajah", "🔢 Klasifikasi Angka"])
st.sidebar.markdown("---")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=120)

# =====================================
# 🏠 HALAMAN COVER
# =====================================
if page == "🏠 Cover":
    st.markdown("""
        <h1 style='text-align:center; color:#00FFFF; text-shadow:0 0 20px #00FFFF;'>✨ UTS BIG DATA & AI ✨</h1>
        <h3 style='text-align:center; color:#A5D7E8;'>Klasifikasi Ekspresi Wajah & Digit Angka</h3>
        <br><br>
        <div style='text-align:center; font-size:18px;'>
            <b>Nama:</b> Ine Lutfiatul Hanifah<br>
            <b>NIM:</b> (Isi NIM kamu di sini)<br>
            <b>Kelas:</b> Big Data & Artificial Intelligence
        </div>
        <br><br>
        <div style='text-align:center; font-size:16px; color:#B0E0E6;'>
            🌟 Selamat datang di dashboard interaktif! Gunakan sidebar di kiri untuk menjelajahi fitur.
        </div>
    """, unsafe_allow_html=True)

# =====================================
# 🎭 EKSPRESI WAJAH
# =====================================
elif page == "🎭 Ekspresi Wajah":
    st.header("🎭 Deteksi Ekspresi Wajah")
    uploaded_file = st.file_uploader("📤 Unggah Gambar Wajah", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

        img_np = np.array(img)
        img_resized = cv2.resize(img_np, (640, 640))

        try:
            results = face_model.predict(img_resized, conf=0.2, agnostic_nms=True, verbose=False)
            if len(results[0].boxes) == 0:
                raise ValueError("Tidak ada deteksi YOLO")

            annotated = results[0].plot()
            st.image(annotated, caption="📸 Deteksi Wajah (YOLO)", use_container_width=True)

            boxes = results[0].boxes
            best = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
            cls = int(best.cls[0])
            conf = float(best.conf[0])
            label = results[0].names.get(cls, "tidak dikenal").lower()

            emoji = {"senang":"😄","bahagia":"😊","sedih":"😢","marah":"😡","takut":"😱","jijik":"🤢"}.get(label,"🙂")

            st.markdown(f"""
                <div style='background:rgba(0,255,255,0.1); padding:20px; border-radius:15px; text-align:center;'>
                    <h2 style='color:#00FFFF'>{emoji} {label.capitalize()}</h2>
                    <p>Akurasi Deteksi: <b>{conf*100:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

        except:
            # fallback deteksi HaarCascade
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 255), 2)
                st.image(img_np, caption="📸 Mode Cadangan: Haar Cascade", use_container_width=True)
                st.info("✅ Wajah terdeteksi dengan mode cadangan.")
            else:
                st.error("❌ Tidak ada wajah terdeteksi sama sekali.")

# =====================================
# 🔢 KLASIFIKASI ANGKA
# =====================================
elif page == "🔢 Klasifikasi Angka":
    st.header("🔢 Klasifikasi Digit Angka")
    uploaded_digit = st.file_uploader("📤 Unggah Gambar Angka (Tulisan tangan)", type=["jpg", "jpeg", "png"])

    if uploaded_digit:
        img = Image.open(uploaded_digit).convert("L")
        st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

        img_resized = img.resize((28, 28))
        arr = image.img_to_array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))

        pred = digit_model.predict(arr)
        pred_label = int(np.argmax(pred[0]))
        prob = float(np.max(pred[0]))
        parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"

        st.markdown(f"""
            <div style='background:rgba(0,255,255,0.1); padding:20px; border-radius:15px; text-align:center;'>
                <h2 style='color:#00FFFF'>Prediksi Angka: {pred_label}</h2>
                <p>Akurasi: <b>{prob*100:.2f}%</b></p>
                <p>{parity}</p>
            </div>
        """, unsafe_allow_html=True)

# =====================================
# 🌙 FOOTER
# =====================================
st.markdown("<br><center style='color:#B0E0E6;'>© 2025 Ine Lutfiatul Hanifah | UTS Big Data & AI 🌙</center>", unsafe_allow_html=True)
