import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from streamlit_extras.let_it_rain import rain

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="UTS Dashboard – Ine Lutfia", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .title { text-align:center; font-size:38px; color:#4169E1; font-weight:bold; }
    .subtitle { text-align:center; font-size:18px; color:gray; margin-top:-15px; }
    .result-box {
        background-color:#F3F6FF; padding:20px; border-radius:20px; text-align:center;
        box-shadow:0px 2px 10px rgba(0,0,0,0.1); margin-top:15px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# STATE NAVIGASI
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Cover"

def goto(page):
    st.session_state.page = page

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    face_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    digit_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    if not os.path.exists(face_path):
        st.error("❌ Model ekspresi (.pt) tidak ditemukan.")
        st.stop()
    if not os.path.exists(digit_path):
        st.error("❌ Model digit (.h5) tidak ditemukan.")
        st.stop()

    face_model = YOLO(face_path)
    digit_model = tf.keras.models.load_model(digit_path)
    return face_model, digit_model

face_model, digit_model = load_models()

# ==========================
# HALAMAN COVER
# ==========================
if st.session_state.page == "Cover":
    st.markdown("<div class='title'>🎓 Dashboard UTS – Big Data & AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Dibuat oleh <b>Ine Lutfia</b> | Universitas Syiah Kuala</div>", unsafe_allow_html=True)
    st.image("LOGO USK.png", width=200)
    rain(emoji="💡", font_size=35, falling_speed=5, animation_length="infinite")
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("😃 Deteksi Ekspresi Wajah"):
            goto("Face Detection")
    with col2:
        if st.button("🔢 Klasifikasi Angka"):
            goto("Digit Classifier")

# ==========================
# HALAMAN EKSPRESI WAJAH
# ==========================
elif st.session_state.page == "Face Detection":
    st.markdown("<h2>🧠 Deteksi Ekspresi Wajah</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📸 Unggah gambar wajah Anda", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar Input", use_container_width=True)

        try:
            results = face_model(img)
            annotated = results[0].plot()
            st.image(annotated, caption="📸 Hasil Deteksi Wajah", use_container_width=True)

            if len(results[0].boxes) == 0:
                st.warning("⚠️ Tidak ada wajah terdeteksi.")
            else:
                labels = results[0].names
                emoji_map = {
                    "senang": "😄", "bahagia": "😊", "sedih": "😢",
                    "marah": "😡", "takut": "😱", "jijik": "🤢"
                }

                # Ambil hasil dengan confidence tertinggi
                best_conf = 0
                best_label = "tidak dikenali"
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_label = labels[int(box.cls[0])].lower()

                # Pesan motivasi sesuai ekspresi
                deskripsi_map = {
                    "senang": "Kamu terlihat bahagia hari ini, teruskan energi positifnya ya! 🌞",
                    "bahagia": "Senyummu menular, tetap semangat dan sebarkan kebaikan! ✨",
                    "sedih": "Jangan khawatir, setiap badai pasti berlalu. 💙",
                    "marah": "Tarik napas dulu ya... kadang hal kecil bisa kita maafkan. 🌿",
                    "takut": "Tenang, kamu lebih kuat dari yang kamu kira. 💪",
                    "jijik": "Mungkin itu bikin nggak nyaman, tapi kamu tetap keren kok. 😅",
                }
                deskripsi = deskripsi_map.get(best_label, "Ekspresimu unik! Terus tampil apa adanya. 💫")

                emoji = emoji_map.get(best_label, "🙂")

                st.markdown(f"""
                    <div class='result-box'>
                        <h2>{emoji} Ekspresi: <b>{best_label.capitalize()}</b></h2>
                        <p style="font-size:17px;">{deskripsi}</p>
                        <p>🎯 Keyakinan: {best_conf*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

                # Efek animasi sesuai emosi
                if best_label in ["senang", "bahagia"]:
                    rain(emoji="✨", font_size=25, falling_speed=5, animation_length="infinite")
                elif best_label == "sedih":
                    rain(emoji="💧", font_size=20, falling_speed=6, animation_length="infinite")
                elif best_label == "marah":
                    rain(emoji="🔥", font_size=25, falling_speed=4, animation_length="infinite")
                elif best_label == "takut":
                    rain(emoji="😨", font_size=25, falling_speed=5, animation_length="infinite")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat deteksi ekspresi: {e}")

    else:
        st.info("⬆️ Unggah gambar wajah untuk mendeteksi ekspresi.")

    st.button("➡️ Lanjut ke Klasifikasi Angka", on_click=lambda: goto("Digit Classifier"))
    st.button("⬅️ Kembali ke Cover", on_click=lambda: goto("Cover"))

# ==========================
# HALAMAN KLASIFIKASI ANGKA
# ==========================
elif st.session_state.page == "Digit Classifier":
    st.markdown("<h2>🔢 Klasifikasi Angka Tulisan Tangan</h2>", unsafe_allow_html=True)
    uploaded_digit = st.file_uploader("📸 Upload gambar angka tulisan tangan", type=["jpg", "jpeg", "png"])

    if uploaded_digit is not None:
        img = Image.open(uploaded_digit).convert('L')
        img = img.resize((28, 28))
        arr = np.array(img).astype("float32") / 255.0

        input_shape = digit_model.input_shape

        try:
            # Auto fix input shape
            if len(input_shape) == 4:
                arr = np.expand_dims(arr, axis=(0, -1))
            elif len(input_shape) == 3:
                arr = np.expand_dims(arr, axis=0)
            else:
                raise ValueError(f"Struktur input tidak dikenali: {input_shape}")

            pred = digit_model.predict(arr)
            angka = int(np.argmax(pred))
            prob = float(np.max(pred))
            parity = "✅ GENAP" if angka % 2 == 0 else "⚠️ GANJIL"

            st.image(img, caption="🖼️ Gambar (Preprocessed)", width=150)
            st.markdown(f"""
                <div class='result-box'>
                    <h2>Angka: {angka}</h2>
                    <h4>Akurasi: {prob*100:.2f}%</h4>
                    <p>{parity}</p>
                </div>
            """, unsafe_allow_html=True)

            # Efek tambahan untuk angka genap/ganjil
            if angka % 2 == 0:
                rain(emoji="💫", font_size=20, falling_speed=5, animation_length="infinite")
            else:
                rain(emoji="🎈", font_size=20, falling_speed=6, animation_length="infinite")

        except Exception as e:
            st.error(f"🚨 Terjadi kesalahan prediksi: {e}")

    else:
        st.info("⬆️ Upload gambar angka untuk klasifikasi")

    st.button("⬅️ Kembali ke Ekspresi", on_click=lambda: goto("Face Detection"))
    st.button("➡️ Tentang AI-ku", on_click=lambda: goto("About"))

# ==========================
# HALAMAN TENTANG
# ==========================
elif st.session_state.page == "About":
    st.markdown("<h2>🤖 Tentang Aplikasi AI-ku</h2>", unsafe_allow_html=True)
    st.markdown("""
        Dashboard ini dibuat sebagai proyek **Ujian Tengah Semester (UTS)** mata kuliah **Big Data & Artificial Intelligence**.  
        Aplikasi ini menggabungkan dua kemampuan AI:
        - 🧠 *Deteksi ekspresi wajah* menggunakan model YOLOv8
        - 🔢 *Klasifikasi angka tulisan tangan* menggunakan model CNN (TensorFlow)

        Fitur unggulan:
        - Desain interaktif dan animasi per ekspresi 🎨  
        - Kalimat motivasional yang berubah sesuai hasil deteksi 💬  
        - Navigasi seperti slide presentasi 📊  
        - Anti-crash: auto fix input shape 🔧  
    """)
    rain(emoji="✨", font_size=25, falling_speed=6, animation_length="infinite")
    st.button("⬅️ Kembali ke Cover", on_click=lambda: goto("Cover"))
