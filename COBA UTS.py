import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import cv2
import os

# ========================================
# 💠 CONFIGURASI AWAL
# ========================================
st.set_page_config(page_title="Deteksi Wajah & Angka", layout="centered")

st.markdown("""
    <style>
    .glass-box {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    .neon-text {
        color: #00FFCC;
        font-weight: bold;
        text-shadow: 0 0 10px #00FFCC, 0 0 20px #00FFCC;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# 🔹 LOAD MODEL
# ========================================
try:
    face_model = tf.keras.models.load_model("face_emotion_model.h5")
    digit_model = tf.keras.models.load_model("INELUTFIATULHANIFAH_LAPORAN 2.h5")
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Model tidak ditemukan: {e}")
    st.stop()

# ========================================
# 🔸 MENU
# ========================================
menu = st.sidebar.selectbox("Pilih Mode", ["Ekspresi Wajah", "Digit Angka"])
show_debug = st.sidebar.checkbox("Tampilkan debug info")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # ========================================
    # 🧠 DETEKSI EKSPRESI WAJAH
    # ========================================
    if menu == "Ekspresi Wajah":
        st.subheader("😊 Hasil Deteksi Ekspresi Wajah")
        try:
            # Pastikan wajah bisa dideteksi
            img_cv = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                st.warning("⚠️ Tidak ada wajah terdeteksi, memproses seluruh gambar sebagai wajah.")
                face_crop = img_cv
            else:
                (x, y, w, h) = faces[0]
                face_crop = img_cv[y:y+h, x:x+w]

            # Resize sesuai model
            h_model, w_model = face_model.input_shape[1:3]
            face_resized = cv2.resize(face_crop, (w_model, h_model))
            face_arr = np.expand_dims(face_resized / 255.0, axis=0)

            # Prediksi
            preds = face_model.predict(face_arr)
            emotion_idx = int(np.argmax(preds))
            emotion_conf = float(np.max(preds))

            labels = ["Marah", "Jijik", "Takut", "Senang", "Sedih", "Kaget", "Netral"]
            emotion_label = labels[emotion_idx] if emotion_idx < len(labels) else "Tidak diketahui"

            st.markdown(f"""
                <div class='glass-box'>
                    <h2 class='neon-text'>{emotion_label}</h2>
                    <p>Akurasi: <b>{emotion_conf*100:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Kesalahan deteksi wajah: {e}")

    # ========================================
    # 🔢 DETEKSI ANGKA (FINAL FIX)
    # ========================================
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Angka")
        try:
            input_shape = digit_model.input_shape
            if len(input_shape) == 4:
                _, h, w, c = input_shape
            else:
                h, w, c = 28, 28, 1

            proc = img.convert("L" if c == 1 else "RGB").resize((w, h))
            arr = np.array(proc)

            # Auto invert jika latar putih
            if np.mean(arr) > 127:
                arr = 255 - arr

            arr = arr.astype("float32") / 255.0

            # Pastikan channel sesuai
            if c == 1 and arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            elif c == 3 and arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)

            arr = np.expand_dims(arr, axis=0)

            if show_debug:
                st.write("📏 Shape input:", arr.shape)

            pred = digit_model.predict(arr)
            angka_pred = int(np.argmax(pred))
            prob = float(np.max(pred))

            if prob < 0.5:
                st.warning("⚠️ Kepercayaan rendah, gambar mungkin kurang jelas.")

            parity = "✅ GENAP" if angka_pred % 2 == 0 else "⚠️ GANJIL"

            st.markdown(f"""
                <div class='glass-box'>
                    <h2 class='neon-text'>Angka: {angka_pred}</h2>
                    <p>Akurasi: <b>{prob*100:.2f}%</b></p>
                    <p>{parity}</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Kesalahan saat klasifikasi digit: {e}")
