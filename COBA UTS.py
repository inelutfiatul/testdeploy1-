import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# =========================
# SETTING DASAR APLIKASI
# =========================
st.set_page_config(page_title="UTS AI Dashboard", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #fdfbfb, #ebedee);
    }
    .title {
        font-size: 42px; 
        font-weight: 800; 
        color: #2c3e50;
        text-align: center;
        letter-spacing: 1px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# NAVIGASI / SLIDE HALAMAN
# =========================
st.sidebar.title("🌈 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Cover", "😊 Deteksi Ekspresi", "🔢 Klasifikasi Angka"])

# =========================
# HALAMAN 1: COVER
# =========================
if page == "🏠 Cover":
    st.markdown("<div class='title'>✨ DASHBOARD KLASIFIKASI CERDAS ✨</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Projek UTS • Deteksi Ekspresi & Klasifikasi Angka</div>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.imgur.com/QYyYz1U.gif", caption="Artificial Intelligence in Action", use_container_width=True)
    with col2:
        st.write("""
        ### 🤖 Selamat Datang!
        Dashboard ini merupakan projek **UTS Big Data & AI** yang berisi dua fitur utama:
        1. 😄 **Deteksi Ekspresi Wajah** — mengenali emosi manusia seperti *senang, sedih, marah, takut, atau jijik*.  
        2. 🔢 **Klasifikasi Angka Tulisan Tangan** — mengenali angka dari gambar dan menentukan apakah **Genap** atau **Ganjil**.  

        Semua ini menggunakan **Deep Learning Model (CNN)** yang dijalankan langsung di Streamlit!  
        """)
    st.markdown("---")
    st.success("Gunakan menu di sidebar ➡️ untuk berpindah halaman!")

# =========================
# HALAMAN 2: DETEKSI EKSPRESI
# =========================
elif page == "😊 Deteksi Ekspresi":
    st.markdown("<div class='title'>😊 Deteksi Ekspresi Wajah</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Unggah gambar wajahmu dan lihat hasil analisis emosi AI!</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📸 Upload Gambar Wajah", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)

        # model ekspresi wajah (pastikan sudah ada di folder)
        try:
            face_model = load_model("face_emotion_model.h5")
        except:
            st.error("❌ Model ekspresi wajah tidak ditemukan! Harap pastikan file 'face_emotion_model.h5' ada.")
            st.stop()

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (48, 48)) / 255.0
        arr = gray.reshape(1, 48, 48, 1)

        pred = face_model.predict(arr)
        label_map = ['marah', 'jijik', 'takut', 'senang', 'sedih', 'bahagia']
        label = label_map[np.argmax(pred)]
        conf = np.max(pred)

        emoji_map = {
            "senang": "😊",
            "bahagia": "😄",
            "sedih": "😢",
            "marah": "😡",
            "takut": "😨",
            "jijik": "🤢"
        }
        text_map = {
            "senang": "Kamu terlihat bahagia hari ini! 🌈",
            "bahagia": "Senyummu menular, terus semangat ya! ☀️",
            "sedih": "Tak apa sedih sebentar, besok pasti cerah lagi. 💙",
            "marah": "Tenang... jangan terbawa emosi ya. 🌿",
            "takut": "Berani menghadapi ketakutan adalah langkah besar! 💪",
            "jijik": "Mungkin hal itu tidak nyaman, tapi kamu tetap hebat 😅"
        }

        emoji = emoji_map.get(label, "🙂")
        pesan = text_map.get(label, "Ekspresimu unik dan menarik! ✨")

        st.image(img, caption="Gambar yang Diuji", use_container_width=True)
        st.markdown(f"""
        <div class='result-box'>
            <h2>{emoji} Ekspresi: <b>{label.capitalize()}</b></h2>
            <p>{pesan}</p>
            <p>🎯 Keyakinan: <b>{conf*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# HALAMAN 3: KLASIFIKASI ANGKA
# =========================
elif page == "🔢 Klasifikasi Angka":
    st.markdown("<div class='title'>🔢 Klasifikasi Angka Tulisan Tangan</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Unggah gambar angka dan lihat hasil prediksi AI!</div>", unsafe_allow_html=True)

    uploaded_digit = st.file_uploader("📤 Upload Gambar Angka (hitam di latar putih)", type=["jpg", "jpeg", "png"])

    if uploaded_digit is not None:
        img = Image.open(uploaded_digit).convert("L")
        img_resized = img.resize((28, 28))
        arr = np.array(img_resized) / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        try:
            digit_model = load_model("digit_model.h5")
        except:
            st.error("❌ Model digit angka tidak ditemukan! Harap pastikan file 'digit_model.h5' ada.")
            st.stop()

        pred = digit_model.predict(arr)
        angka = np.argmax(pred)
        akurasi = np.max(pred)

        hasil = "✅ Genap" if angka % 2 == 0 else "🔥 Ganjil"

        st.image(img, caption="Gambar yang Diuji", use_container_width=True)
        st.markdown(f"""
        <div class='result-box'>
            <h2>📍 Angka Terdeteksi: <b>{angka}</b></h2>
            <h3>{hasil}</h3>
            <p>🎯 Keyakinan: <b>{akurasi*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 – Ine Lutfia • UTS Big Data & AI 💡</p>", unsafe_allow_html=True)
