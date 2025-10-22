import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==========================
# CONFIG & STYLE
# ==========================
st.set_page_config(page_title="Klasifikasi Ekspresi & Digit", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 32px;
        color: #4B7BE5;
        font-weight: bold;
    }
    .subheader {
        color: #333;
        font-size: 20px;
        text-align: center;
        margin-top: -10px;
    }
    .result-box {
        background-color: #F5F7FF;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    face_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    digit_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    if not os.path.exists(face_path):
        st.error("❌ File model ekspresi wajah (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(digit_path):
        st.error("❌ File model digit angka (.h5) tidak ditemukan.")
        st.stop()

    face_model = YOLO(face_path)
    digit_model = tf.keras.models.load_model(digit_path)
    return face_model, digit_model


# ✅ Load model
face_model, digit_model = load_models()

# ==========================
# UI HEADER
# ==========================
st.markdown("<div class='title'>🧠 Dashboard Klasifikasi Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Proyek UAS – Big Data & AI</div>", unsafe_allow_html=True)
st.write("")

logo_path = "LOGO USK.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
else:
    st.sidebar.warning("⚠️ Logo tidak ditemukan")

st.sidebar.header("⚙️ Pengaturan")
menu = st.sidebar.radio("Pilih Jenis Klasifikasi:", ["Ekspresi Wajah", "Digit Angka"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROCESSING
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # ===================================
# 1️⃣ EKSPRESI WAJAH (.pt)
# ===================================
if menu == "Ekspresi Wajah":
    st.subheader("🔍 Hasil Deteksi Ekspresi Wajah")

    try:
        # Jalankan deteksi YOLO
        results = face_model(img)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="📸 Deteksi Ekspresi", use_container_width=True)

        # Jika tidak ada wajah terdeteksi
        if len(results[0].boxes) == 0:
            st.warning("⚠️ Tidak ada wajah terdeteksi. Silakan unggah gambar dengan wajah yang jelas.")
        else:
            # Daftar label ekspresi (urutannya harus sama dengan model kamu)
            ekspresi_labels = ["senang", "sedih", "marah", "takut", "jijik"]

            # Emoji per ekspresi biar menarik 😄
            emoji_map = {
                "senang": "😄",
                "sedih": "😢",
                "marah": "😡",
                "takut": "😱",
                "jijik": "🤢"
            }

            for box in results[0].boxes:
                cls = int(box.cls[0]) if box.cls is not None else 0
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                # Ambil label dari daftar (pastikan index tidak keluar batas)
                if 0 <= cls < len(ekspresi_labels):
                    label = ekspresi_labels[cls]
                else:
                    label = "Tidak Dikenal"

                emoji = emoji_map.get(label, "🙂")

                st.markdown(
                    f"""
                    <div style="
                        background-color:#f8f9fa;
                        border-radius:12px;
                        padding:16px;
                        margin-top:10px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                        text-align:center;
                    ">
                        <h3>Ekspresi: {emoji} <b>{label.capitalize()}</b></h3>
                        <p style="font-size:16px;">Keyakinan: <b>{conf*100:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat deteksi ekspresi: {e}")

    # ===================================
    # 2️⃣ DIGIT ANGKA (.h5)
    # ===================================
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Digit Angka")
        try:
            # --- AUTO DETEKSI INPUT MODEL ---
            input_shape = digit_model.input_shape
            target_size = (input_shape[1], input_shape[2])
            channels = input_shape[3]

            # --- SESUAIKAN WARNA DENGAN JUMLAH CHANNEL ---
            if channels == 1:
                img_proc = img.convert("L")
            else:
                img_proc = img.convert("RGB")

            # --- RESIZE SESUAI MODEL ---
            img_resized = img_proc.resize(target_size)
            img_array = image.img_to_array(img_resized)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # --- PREDIKSI ---
            pred = digit_model.predict(img_array)
            pred_label = int(np.argmax(pred))
            prob = float(np.max(pred))

            colA, colB = st.columns(2)
            with colA:
                st.image(img_resized, caption="🖼️ Gambar Uji", use_container_width=True)
            with colB:
                parity = "✅ GENAP" if pred_label % 2 == 0 else "⚠️ GANJIL"
                st.markdown(f"""
                    <div class='result-box'>
                        <h2>Angka: {pred_label}</h2>
                        <h4>Akurasi: {prob:.2%}</h4>
                        <p>{parity}</p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat klasifikasi digit: {e}")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk melakukan deteksi atau klasifikasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 – Dibuat oleh Ine Lutfia • Proyek UAS Big Data</p>", unsafe_allow_html=True)
