import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import cv2
import os

# ================================
# 🌌 CONFIG + STYLE
# ================================
st.set_page_config(page_title="AI Dashboard UTS", page_icon="🤖", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    text-align: center;
    color: #A5D7E8;
    text-shadow: 0 0 10px #00FFFF;
}
.glass-box {
    background: rgba(255,255,255,0.1);
    border-radius: 25px;
    padding: 25px;
    box-shadow: 0 0 30px rgba(0,255,255,0.2);
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease-in-out;
}
.glass-box:hover {
    transform: scale(1.02);
    box-shadow: 0 0 40px #00FFFF;
}
.footer {
    text-align: center;
    color: #B0E0E6;
    font-size: 13px;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ================================
# ⚙️ LOAD MODEL
# ================================
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

# ================================
# 🧭 SIDEBAR MENU
# ================================
st.sidebar.header("🌠 Navigasi Dashboard")
menu = st.sidebar.radio("Pilih Mode:", ["🎭 Ekspresi Wajah", "🔢 Klasifikasi Angka"])
st.sidebar.markdown("---")
st.sidebar.info("✨ Proyek UTS – Big Data & AI")

uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ================================
# 🎭 DETEKSI EKSPRESI WAJAH
# ================================
if menu == "🎭 Ekspresi Wajah":
    st.markdown("<h2>🧠 Deteksi Ekspresi Wajah</h2>", unsafe_allow_html=True)

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📷 Gambar Input", use_container_width=True)

        # === AUTO ENHANCE ===
        def enhance_image(img_pil):
            enhancer1 = ImageEnhance.Contrast(img_pil)
            enhancer2 = ImageEnhance.Brightness(enhancer1.enhance(1.5))
            enhancer3 = ImageEnhance.Sharpness(enhancer2.enhance(1.3))
            return enhancer3.enhance(1.2)

        enhanced_img = enhance_image(img)
        cv_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)

        try:
            # === YOLO DETECTION ===
            results = face_model(enhanced_img)

            # === jika gagal deteksi, lakukan recovery otomatis ===
            if len(results[0].boxes) == 0:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    st.warning("⚙️ Wajah tidak terdeteksi otomatis, sistem menggunakan fallback Haar Cascade.")
                else:
                    st.warning("⚙️ Tidak ada wajah yang jelas, tapi sistem akan tetap mengklasifikasi ekspresi umum.")

            annotated_img = results[0].plot()
            st.image(annotated_img, caption="📍 Hasil Deteksi", use_container_width=True)

            # === Ambil hasil terbaik ===
            boxes = results[0].boxes
            if len(boxes) == 0:
                label = "tidak terdeteksi"
                conf = 0.0
            else:
                best_box = boxes[np.argmax([float(b.conf[0]) for b in boxes])]
                cls = int(best_box.cls[0])
                conf = float(best_box.conf[0])
                label = results[0].names.get(cls, "Tidak Dikenal").lower()

            emoji_map = {
                "senang": "😄", "bahagia": "😊", "sedih": "😢",
                "marah": "😡", "takut": "😱", "jijik": "🤢"
            }
            emoji = emoji_map.get(label, "🙂")

            st.markdown(f"""
                <div class='glass-box'>
                    <h2>{emoji} {label.capitalize()}</h2>
                    <p>Akurasi Deteksi: <b>{conf*100:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat deteksi wajah: {e}")
    else:
        st.info("⬆️ Upload gambar wajah terlebih dahulu untuk mulai deteksi.")

# ================================
# 🔢 KLASIFIKASI ANGKA
# ================================
elif menu == "🔢 Klasifikasi Angka":
    st.markdown("<h2>🔢 Klasifikasi Angka Tulisan Tangan</h2>", unsafe_allow_html=True)

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        st.image(img, caption="🖼️ Gambar Input", width=200)

        try:
            img_array = np.array(img)
            img_array = cv2.resize(img_array, (28, 28))

            if np.mean(img_array) > 127:
                img_array = 255 - img_array

            img_array = img_array.astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)

            pred = digit_model.predict(img_array)
            angka = int(np.argmax(pred))
            prob = float(np.max(pred))
            parity = "✅ GENAP" if angka % 2 == 0 else "⚠️ GANJIL"

            st.markdown(f"""
            <div class='glass-box'>
                <h2>🎯 Hasil Prediksi: {angka}</h2>
                <p>Akurasi: <b>{prob*100:.2f}%</b></p>
                <p>{parity}</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Kesalahan saat klasifikasi angka: {e}")
    else:
        st.info("⬆️ Upload gambar angka terlebih dahulu untuk mulai klasifikasi.")

# ================================
# 🌙 FOOTER
# ================================
st.markdown("<div class='footer'>© 2025 – Ine Lutfia | Dashboard UTS Big Data & AI ✨</div>", unsafe_allow_html=True)
