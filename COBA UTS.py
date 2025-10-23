import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# =========================
# SETTING DASAR APLIKASI
# =========================
st.set_page_config(page_title="UTS AI Dashboard", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa, #fce4ec);
    }
    .title {
        font-size: 45px; 
        font-weight: 700; 
        color: #2c3e50;
        text-align: center;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #34495e;
    }
    .result-box {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# NAVIGASI UNIK (SLIDE)
# =========================
st.sidebar.title("🌈 Navigasi Halaman")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Cover", "😊 Deteksi Ekspresi Wajah", "🔢 Klasifikasi Angka"])

# =========================
# HALAMAN 1: COVER
# =========================
if page == "🏠 Cover":
    st.markdown("<div class='title'>DASHBOARD KLASIFIKASI CERDAS 🤖</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Projek UTS • Deteksi Ekspresi & Angka Digital</div>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.imgur.com/MHAdK1y.gif", caption="AI & Deep Learning", use_container_width=True)
    with col2:
        st.write("""
        ### 👋 Selamat Datang!
        Dashboard ini dibuat sebagai **projek UTS** yang memadukan dua fitur kecerdasan buatan:
        1. 😄 **Deteksi Ekspresi Wajah** — mengenali emosi manusia seperti senang, sedih, marah, takut, atau jijik.  
        2. 🔢 **Klasifikasi Angka Tulisan Tangan** — mengenali angka dari gambar, lalu mengelompokkannya menjadi **genap** atau **ganjil**.  

        Semua ini berjalan menggunakan **model deep learning (CNN)** yang diproses langsung di Streamlit.
        """)
    st.markdown("---")
    st.success("Klik menu di sidebar ➡️ untuk mulai menggunakan fitur!")

# =========================
# HALAMAN 2: DETEKSI EKSPRESI WAJAH
# =========================
elif page == "😊 Deteksi Ekspresi Wajah":
    st.markdown("<div class='title'>Deteksi Ekspresi Wajah 😃</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Unggah foto wajahmu untuk mendeteksi emosi!</div>", unsafe_allow_html=True)

    face_model = load_model("face_emotion_model.h5")  # pastikan model ini ada
    uploaded_file = st.file_uploader("📸 Upload Gambar Wajah", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48)) / 255.0
        arr = resized.reshape(1, 48, 48, 1)

        pred = face_model.predict(arr)
        label_map = ['marah', 'jijik', 'takut', 'senang', 'sedih', 'bahagia']
        best_label = label_map[np.argmax(pred)]
        best_conf = np.max(pred)

        emoji_map = {
            "senang": "😊",
            "bahagia": "😄",
            "sedih": "😢",
            "marah": "😡",
            "takut": "😨",
            "jijik": "🤢"
        }
        deskripsi_map = {
            "senang": "Kamu terlihat bahagia hari ini, teruskan energi positifnya ya! 🌞",
            "bahagia": "Senyummu menular, tetap semangat dan sebarkan kebaikan! ✨",
            "sedih": "Jangan khawatir, setiap badai pasti berlalu. 💙",
            "marah": "Tarik napas dulu ya... kadang hal kecil bisa kita maafkan. 🌿",
            "takut": "Tenang, kamu lebih kuat dari yang kamu kira. 💪",
            "jijik": "Mungkin itu bikin nggak nyaman, tapi kamu tetap keren kok. 😅",
        }

        emoji = emoji_map.get(best_label, "🙂")
        deskripsi = deskripsi_map.get(best_label, "Ekspresimu unik! Terus tampil apa adanya. 💫")

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar Diuji", use_container_width=True)
        st.markdown(f"""
            <div class='result-box'>
                <h2>{emoji} Ekspresi: <b>{best_label.capitalize()}</b></h2>
                <p style="font-size:17px;">{deskripsi}</p>
                <p>🎯 Keyakinan: {best_conf*100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

# =========================
# HALAMAN 3: KLASIFIKASI ANGKA
# =========================
elif page == "🔢 Klasifikasi Angka":
    st.markdown("<div class='title'>Klasifikasi Angka Tulisan Tangan ✍️</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Gambarlah angka, lalu sistem akan mengenali dan menentukan apakah itu Genap atau Ganjil!</div>", unsafe_allow_html=True)

    digit_model = load_model("digit_model.h5")  # pastikan model ada

    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("🔍 Deteksi Angka"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img = cv2.resize(img, (28, 28))
            img = cv2.dilate(img, np.ones((2,2), np.uint8), iterations=1)
            img = cv2.erode(img, np.ones((2,2), np.uint8), iterations=1)
            img = 255 - img  # inversi warna
            img = img / 255.0
            arr = img.reshape(1, 28, 28, 1)

            pred = digit_model.predict(arr)
            angka = np.argmax(pred)

            st.markdown(f"<div class='result-box'><h2>🔢 Angka Terdeteksi: <b>{angka}</b></h2></div>", unsafe_allow_html=True)
            
            if angka % 2 == 0:
                st.success(f"✨ Angka {angka} termasuk **Genap**")
            else:
                st.warning(f"🔥 Angka {angka} termasuk **Ganjil**")
        else:
            st.error("Gambarlah angka terlebih dahulu di kanvas!")

