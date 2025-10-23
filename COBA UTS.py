import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==========================
# 🌌 CONFIG & STYLE
# ==========================
st.set_page_config(page_title="AI Klasifikasi Ekspresi & Digit", page_icon="🤖", layout="wide")

st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0F2027, #203A43, #2C5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    text-align: center; 
    font-size: 38px; 
    font-weight: 800; 
    color: #A5D7E8;
    text-shadow: 0px 0px 10px #00FFFF;
}
.subheader {
    text-align: center; 
    font-size: 18px; 
    color: #D9EAFD;
    margin-top: -10px;
}
.glass-box {
    background: rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 4px 25px rgba(0,255,255,0.15);
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease-in-out;
}
.glass-box:hover {
    box-shadow: 0 0 25px #00FFFF;
    transform: scale(1.02);
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
# ⚡ MAIN PROCESS
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # 1️⃣ EKSPRESI WAJAH
    if menu == "Ekspresi Wajah":
        st.subheader("🎭 Hasil Deteksi Ekspresi Wajah")
        try:
            results = face_model(img)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="📸 Deteksi Wajah", use_container_width=True)

            if len(results[0].boxes) == 0:
                st.warning("😅 Tidak ada wajah terdeteksi. Gunakan foto wajah yang jelas.")
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

                # Respon AI mini
                if label in ["sedih", "takut"]:
                    st.info("💬 Aku harap kamu baik-baik aja. Jangan lupa senyum hari ini ya! 😊")
                elif label in ["bahagia", "senang"]:
                    st.success("💬 Wah, senyummu menular banget! Terus semangat ya 😄✨")
                elif label == "marah":
                    st.warning("💬 Tenang dulu ya... ambil napas dalam 🧘‍♂️")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat deteksi wajah: {e}")

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
                st.write("Prediksi mentah:", pred)

        except Exception as e:
            st.error(f"❌ Kesalahan saat klasifikasi digit: {e}")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk mulai klasifikasi.")

# ==========================
# 🌙 FOOTER
# ==========================
st.markdown("<div class='footer'>© 2025 – Ine Lutfia | Proyek UTS Big Data & AI ✨</div>", unsafe_allow_html=True)
