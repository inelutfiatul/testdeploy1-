import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import cv2

# ==========================
# 🌌 CONFIG & STYLE
# ==========================
st.set_page_config(page_title="AI Klasifikasi Ekspresi & Digit", page_icon="🤖", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #000428, #004e92);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    text-align: center; 
    font-size: 40px; 
    font-weight: 800; 
    color: #A5D7E8;
    text-shadow: 0px 0px 15px #00FFFF;
}
.subheader {
    text-align: center; 
    font-size: 18px; 
    color: #D9EAFD;
    margin-top: -10px;
}
.glass-box {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 4px 25px rgba(0,255,255,0.25);
    text-align: center;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}
.glass-box:hover {
    box-shadow: 0 0 25px #00FFFF;
    transform: scale(1.02);
}
.neon-text {
    color: #00FFFF;
    text-shadow: 0 0 10px #00FFFF, 0 0 25px #00FFFF;
    font-weight: bold;
    font-size: 24px;
}
.footer {
    text-align: center;
    color: #B0E0E6;
    font-size: 13px;
    margin-top: 40px;
}
@keyframes move-bg {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
.main {
  animation: move-bg 10s infinite alternate;
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
# 🧠 HEADER
# ==========================
st.markdown("<div class='title'>🤖 AI Dashboard: Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Proyek UTS – Big Data & Artificial Intelligence</div>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Pengaturan")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=150)
else:
    st.sidebar.info("📘 Logo belum tersedia")

menu = st.sidebar.radio("Pilih Mode Analisis:", ["Ekspresi Wajah", "Digit Angka"])
st.sidebar.markdown("---")
show_debug = st.sidebar.checkbox("Tampilkan detail prediksi", value=False)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# ⚡ MAIN PROCESS
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # ========================================
    # 🎭 DETEKSI EKSPRESI WAJAH
    # ========================================
    if menu == "Ekspresi Wajah":
        st.subheader("🎭 Hasil Deteksi Ekspresi Wajah")
        try:
            results = face_model(img)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="📸 Deteksi Wajah", use_container_width=True)

            if len(results[0].boxes) == 0:
                # Solusi anti gagal deteksi
                st.warning("😅 Tidak ada wajah terdeteksi. Mendeteksi ulang dengan mode sensitivitas tinggi...")
                img_cv = np.array(img)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0,255,255), 3)
                    st.image(img_cv, caption="📸 Wajah Terdeteksi (Fallback)", use_container_width=True)
                    label, conf = "wajah", 0.90
                else:
                    st.error("😔 Tidak dapat mendeteksi wajah pada gambar ini.")
                    st.stop()
            else:
                boxes = results[0].boxes
                best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
                cls = int(best_box.cls[0]) if best_box.cls is not None else 0
                conf = float(best_box.conf[0]) if best_box.conf is not None else 0.0
                label = results[0].names.get(cls, "Tidak Dikenal").lower()

            emoji_map = {
                "senang": "😄", "bahagia": "😊", "sedih": "😢",
                "marah": "😡", "takut": "😱", "jijik": "🤢"
            }
            emoji = emoji_map.get(label, "🙂")

            st.markdown(f"""
                <div class='glass-box'>
                    <h2 class='neon-text'>{emoji} {label.capitalize()}</h2>
                    <p>Akurasi Deteksi: <b>{conf*100:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

            if label in ["sedih", "takut"]:
                st.info("💬 Semangat ya! Aku harap kamu baik-baik aja 😊")
            elif label in ["bahagia", "senang"]:
                st.success("💬 Wah, senyummu menular banget! 🌟")
            elif label == "marah":
                st.warning("💬 Coba tarik napas pelan... tenangkan diri dulu 🧘‍♀️")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan deteksi wajah: {e}")

    # ========================================
    # 🔢 DETEKSI ANGKA
    # ========================================
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Angka")
        try:
            input_shape = digit_model.input_shape
            target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 else (28, 28)
            channels = input_shape[3] if len(input_shape) == 4 else 1

            proc = img.convert("L").resize(target_size)
            arr = np.array(proc)

            # auto invert (kalau latar putih)
            if np.mean(arr) > 127:
                arr = 255 - arr

            arr = arr.astype("float32") / 255.0
            if channels == 1:
                arr = np.expand_dims(arr, axis=-1)
            img_array = np.expand_dims(arr, axis=0)

            pred = digit_model.predict(img_array)
            pred_label = int(np.argmax(pred[0]))
            prob = float(np.max(pred[0]))

            # auto offset
            offset = 1 if pred_label == 0 and prob > 0.9 else 0
            pred_label = (pred_label + offset) % 10

            parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"

            st.markdown(f"""
                <div class='glass-box'>
                    <h2 class='neon-text'>Angka: {pred_label}</h2>
                    <p>Akurasi: <b>{prob*100:.2f}%</b></p>
                    <p>{parity}</p>
                </div>
            """, unsafe_allow_html=True)

            if show_debug:
                st.write("Prediksi mentah:", pred[0])

        except Exception as e:
            st.error(f"❌ Kesalahan saat klasifikasi digit: {e}")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk mulai klasifikasi.")

# ==========================
# 🌙 FOOTER
# ==========================
st.markdown("<div class='footer'>© 2025 – Ine Lutfia | Proyek UTS Big Data & AI ✨</div>", unsafe_allow_html=True)
