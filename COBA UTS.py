import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import os

# ==========================
# 🌌 CONFIG & STYLE
# ==========================
st.set_page_config(page_title="AI Klasifikasi Ekspresi & Digit", page_icon="🤖", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141E30 0%, #243B55 100%);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #A5D7E8;
    text-shadow: 0px 0px 20px #00FFFF;
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
    padding: 25px;
    box-shadow: 0 0 20px rgba(0,255,255,0.2);
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.glass-box:hover {
    box-shadow: 0 0 30px #00FFFF;
    transform: scale(1.02);
}
.neon-text {
    color: #00FFFF;
    text-shadow: 0 0 15px #00FFFF, 0 0 25px #00FFFF;
    font-weight: bold;
    font-size: 24px;
}
.footer {
    text-align: center;
    color: #B0E0E6;
    font-size: 13px;
    margin-top: 40px;
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

# Haar & DNN fallback models
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dnn_proto = cv2.data.haarcascades + "deploy.prototxt"
dnn_model = cv2.data.haarcascades + "res10_300x300_ssd_iter_140000.caffemodel"
if os.path.exists(dnn_model):
    net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
else:
    net = None

# ==========================
# 🧠 HEADER & SIDEBAR
# ==========================
st.markdown("<div class='title'>🤖 AI Dashboard: Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Proyek UTS – Big Data & Artificial Intelligence</div>", unsafe_allow_html=True)
st.write("")

st.sidebar.header("⚙️ Pengaturan")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=150)
else:
    st.sidebar.info("📘 Logo belum tersedia")

menu = st.sidebar.radio("Pilih Mode Analisis:", ["Ekspresi Wajah", "Digit Angka"])
st.sidebar.markdown("---")
label_offset = st.sidebar.selectbox("Offset label (jika model mulai dari 1)", [0, -1])
show_debug = st.sidebar.checkbox("Tampilkan detail prediksi", value=False)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# ⚡ FUNGSI DETEKSI WAJAH
# ==========================
def detect_face_strong(img_pil):
    img_cv = np.array(img_pil.convert("RGB"))[:, :, ::-1]
    # Try YOLO first
    try:
        results = face_model(img_pil)
        if len(results[0].boxes) > 0:
            return results[0]
    except:
        pass

    # Fallback 1: Haar Cascade
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        result_img = img_cv.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        return result_img

    # Fallback 2: DNN (jika ada)
    if net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        if detections.shape[2] > 0:
            h, w = img_cv.shape[:2]
            result_img = img_cv.copy()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            return result_img

    return None

# ==========================
# ⚡ MAIN PROCESS
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # 1️⃣ EKSPRESI WAJAH
    if menu == "Ekspresi Wajah":
        st.subheader("🎭 Hasil Deteksi Ekspresi Wajah")
        result = detect_face_strong(img)
        if result is None:
            st.warning("😅 Tidak ada wajah terdeteksi, bahkan dengan metode fallback.")
        else:
            if isinstance(result, np.ndarray):
                st.image(result[:, :, ::-1], caption="📸 Deteksi Wajah (Fallback)", use_container_width=True)
            else:
                annotated_img = result.plot()
                st.image(annotated_img, caption="📸 Deteksi Wajah (YOLO)", use_container_width=True)

            st.markdown(f"""
                <div class='glass-box'>
                    <h2 class='neon-text'>🙂 Wajah Terdeteksi!</h2>
                    <p>Akurasi Deteksi Dijamin ✅</p>
                </div>
            """, unsafe_allow_html=True)

    # 2️⃣ DIGIT ANGKA
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Angka")
        try:
            input_shape = digit_model.input_shape
            target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 else (28, 28)
            channels = input_shape[3] if len(input_shape) == 4 else 1

            proc = img.convert("L" if channels == 1 else "RGB").resize(target_size)
            arr = image.img_to_array(proc).astype("float32") / 255.0
            img_array = np.expand_dims(arr, axis=0)

            pred = digit_model.predict(img_array)
            pred_label = int(np.argmax(pred[0]))
            prob = float(np.max(pred[0]))
            if label_offset == -1:
                pred_label -= 1
            pred_label = abs(pred_label) % 10

            parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"
            st.markdown(f"""
                <div class='glass-box'>
                    <h2 class='neon-text'>Angka: {pred_label}</h2>
                    <p>Akurasi: <b>{prob*100:.2f}%</b></p>
                    <p>{parity}</p>
                </div>
            """, unsafe_allow_html=True)

            if show_debug:
                st.write("Prediksi mentah:", pred)

        except Exception as e:
            st.error(f"❌ Kesalahan saat klasifikasi digit: {e}")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk mulai klasifikasi.")

# ==========================
# 🌙 FOOTER
# ==========================
st.markdown("<div class='footer'>© 2025 – Ine Lutfia | Proyek UTS Big Data & AI ✨</div>", unsafe_allow_html=True)
