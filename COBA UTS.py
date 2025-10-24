import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import cv2
import os

# ==============================
# 🌌 CONFIG & STYLE
# ==============================
st.set_page_config(page_title="AI Dashboard UTS – Ine Lutfia", page_icon="🤖", layout="wide")

st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0F2027, #203A43, #2C5364);
    font-family: 'Poppins', sans-serif;
    color: #EAF6F6;
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
    color: #B8E3FF;
    margin-top: -10px;
}
.glass-box {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 4px 25px rgba(0,255,255,0.15);
    text-align: center;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}
.glass-box:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px #00FFFF;
}
.neon-text {
    color: #00FFFF;
    text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF;
    font-weight: bold;
    font-size: 22px;
}
.footer {
    text-align: center;
    color: #B0E0E6;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🚀 LOAD MODELS
# ==============================
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
    haar_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    return face_model, digit_model, haar_model

face_model, digit_model, haar_model = load_models()

# ==============================
# 🧠 UTILITAS: Enhance + Fallback
# ==============================
def enhance_image(img):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    return img

def detect_face_strong(img):
    # 1️⃣ YOLO
    results = face_model(img, conf=0.15)
    if len(results[0].boxes) > 0:
        return results, "YOLO"

    # 2️⃣ Haar Cascade (fallback)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    faces = haar_model.detectMultiScale(img_cv, 1.2, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return [{"boxes": faces, "img": img_cv}], "Haar Cascade"

    # 3️⃣ OpenCV DNN (ultimate fallback)
    prototxt = cv2.data.haarcascades + "deploy.prototxt"
    model = cv2.data.haarcascades + "res10_300x300_ssd_iter_140000.caffemodel"
    if os.path.exists(model):
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        blob = cv2.dnn.blobFromImage(np.array(img), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        if detections.shape[2] > 0:
            return detections, "OpenCV DNN"

    return None, "None"

# ==============================
# 🧩 SIDEBAR & HEADER
# ==============================
st.markdown("<div class='title'>🤖 AI Dashboard – Ekspresi & Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>UTS Big Data & Artificial Intelligence | Ine Lutfia</div>", unsafe_allow_html=True)
st.sidebar.header("⚙️ Navigasi")
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Ekspresi", "Klasifikasi Angka"])

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==============================
# 🎭 DETEKSI EKSPRESI WAJAH
# ==============================
if menu == "Deteksi Ekspresi":
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img = enhance_image(img)
        st.image(img, caption="Gambar Input", use_container_width=True)

        results, method = detect_face_strong(img)

        if results is None:
            st.error("❌ Wajah tidak dapat dideteksi oleh sistem apapun.")
        else:
            st.success(f"✅ Wajah berhasil dideteksi dengan metode: {method}")

            if method == "YOLO":
                annotated = results[0].plot()
                st.image(annotated, caption="📸 Hasil Deteksi Wajah", use_container_width=True)
                best_box = results[0].boxes[0]
                cls = int(best_box.cls[0])
                conf = float(best_box.conf[0])
                label = results[0].names.get(cls, "Tidak dikenal").lower()
            else:
                label = "tidak dikenal"
                conf = 0.8

            emoji = {
                "senang": "😄", "bahagia": "😊", "sedih": "😢",
                "marah": "😡", "takut": "😱", "jijik": "🤢"
            }.get(label, "🙂")

            st.markdown(f"""
                <div class='glass-box'>
                    <h2 class='neon-text'>{emoji} Ekspresi: {label.capitalize()}</h2>
                    <p>Akurasi Deteksi: <b>{conf*100:.2f}%</b></p>
                    <p>Mode Deteksi: <b>{method}</b></p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("⬆️ Unggah gambar wajah untuk mulai deteksi.")

# ==============================
# 🔢 KLASIFIKASI ANGKA
# ==============================
elif menu == "Klasifikasi Angka":
    if uploaded_file:
        img = Image.open(uploaded_file)
        input_shape = digit_model.input_shape
        h, w, c = input_shape[1], input_shape[2], input_shape[3]

        img = img.convert('L' if c == 1 else 'RGB').resize((h, w))
        arr = np.array(img).astype("float32") / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)

        pred = digit_model.predict(arr)
        label = int(np.argmax(pred))
        prob = float(np.max(pred))

        st.image(img, caption="🖼️ Gambar (Preprocessed)", width=150)
        parity = "✅ GENAP" if label % 2 == 0 else "⚠️ GANJIL"
        st.markdown(f"""
            <div class='glass-box'>
                <h2 class='neon-text'>Angka: {label}</h2>
                <p>Akurasi: <b>{prob*100:.2f}%</b></p>
                <p>{parity}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⬆️ Unggah gambar angka tulisan tangan untuk klasifikasi.")

# ==============================
# 🌙 FOOTER
# ==============================
st.markdown("<div class='footer'>© 2025 Ine Lutfia | AI UTS Dashboard – Anti Gagal Deteksi 💫</div>", unsafe_allow_html=True)
