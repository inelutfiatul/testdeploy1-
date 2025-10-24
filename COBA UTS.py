import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2
import io

# ===============================================
# 💛 TEMA GLOWING EMAS ELEGAN
# ===============================================
st.set_page_config(page_title="Deteksi Ekspresi & Angka", page_icon="✨", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(180deg, #000000, #3b2f1d, #d4af37);
        color: white;
    }
    .title {
        font-size: 42px; 
        font-weight: 800; 
        color: #FFD700; 
        text-align: center;
        text-shadow: 0 0 20px #FFD700;
        margin-bottom: 15px;
    }
    .section-title {
        color: #FFF5B7;
        font-size: 26px;
        font-weight: 600;
        text-shadow: 0 0 15px #FFD700;
        margin-top: 25px;
    }
    .result {
        font-size: 22px;
        font-weight: 600;
        color: #FFF8DC;
        text-shadow: 0 0 10px #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>💫 Deteksi Ekspresi Wajah & Angka Ajaib 💫</h1>", unsafe_allow_html=True)
st.markdown("Selamat datang! Unggah foto wajah atau angka, dan biarkan sistem mendeteksi dengan gaya elegan & lucu 😆")

# ===============================================
# 📦 LOAD MODEL
# ===============================================
try:
    face_model = tf.keras.models.load_model("face_emotion_model.h5")
    digit_model = tf.keras.models.load_model("digit_model.h5")
    st.success("✅ Model berhasil dimuat!")
except:
    st.error("❌ Model tidak ditemukan. Pastikan file .h5 tersedia di direktori yang sama.")
    st.stop()

# ===============================================
# 📸 UNGGAH GAMBAR
# ===============================================
uploaded_file = st.file_uploader("Unggah gambar (wajah / angka):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang diunggah", use_container_width=True)

    # Konversi ke array numpy
    img_array = np.array(image)

    # ===========================================
    # 🔹 DETEKSI EKSPRESI WAJAH
    # ===========================================
    if st.button("✨ Deteksi Ekspresi Wajah"):
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                st.warning("😕 Wajah tidak terdeteksi. Coba pencahayaan lebih terang atau posisi depan.")
            else:
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (48, 48))
                    face = face / 255.0
                    face = np.reshape(face, (1, 48, 48, 1))
                    pred = face_model.predict(face)
                    emotion_labels = ['Marah 😡', 'Jijik 🤢', 'Takut 😨', 'Senang 😄', 'Sedih 😢']
                    emotion = emotion_labels[np.argmax(pred)]

                    st.markdown(f"<div class='result'>🧠 Ekspresi terdeteksi: <b>{emotion}</b></div>", unsafe_allow_html=True)

                    # 🪞 Caption lucu otomatis
                    captions = {
                        "Senang 😄": "Hehe… kayaknya kamu lagi senyum-senyum sendiri 😆",
                        "Sedih 😢": "Duh, jangan sedih dong 😢 semangat lagi ya!",
                        "Marah 😡": "Ups, kok keliatan marah ya 😡 santai dulu deh~",
                        "Takut 😨": "Waduh, kenapa takut gitu 😨 tenang aja, aman kok!",
                        "Jijik 🤢": "Hiii... apa sih yang bikin kamu jijik gitu 😅"
                    }
                    st.markdown(f"🌟 <i>{captions.get(emotion, '')}</i>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Kesalahan saat mendeteksi ekspresi: {e}")

    # ===========================================
    # 🔹 DETEKSI ANGKA
    # ===========================================
    if st.button("🔢 Deteksi Angka"):
        try:
            img_resized = image.convert("L").resize((28, 28))
            img_inverted = ImageOps.invert(img_resized)
            arr = np.array(img_inverted).astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=(0, -1))

            pred = digit_model.predict(arr)
            digit = np.argmax(pred)

            st.markdown(f"<div class='result'>📊 Angka terdeteksi: <b>{digit}</b></div>", unsafe_allow_html=True)

            # Genap / Ganjil
            if digit % 2 == 0:
                st.markdown("✨ Angka ini termasuk <b>Genap</b> 🟡", unsafe_allow_html=True)
            else:
                st.markdown("✨ Angka ini termasuk <b>Ganjil</b> 🟢", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Kesalahan saat klasifikasi digit: {e}")

else:
    st.info("📂 Silakan unggah gambar terlebih dahulu untuk mulai mendeteksi.")
