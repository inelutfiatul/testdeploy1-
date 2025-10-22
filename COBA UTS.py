import streamlit as st
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# =============================
# KONFIGURASI DASHBOARD
# =============================
st.set_page_config(
    page_title="Dashboard Klasifikasi AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#6a0dad;'>🧠 Dashboard Klasifikasi AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>By Ine Lutfiatul Hanifah | Statistika | Big Data Project 2025</h4>", unsafe_allow_html=True)
st.markdown("---")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_face_model():
    # model ekspresi wajah (PyTorch)
    model = torch.load("Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

@st.cache_resource
def load_digit_model():
    # model digits (Keras/TensorFlow)
    return load_model("INELUTFIATULHANIFAH_LAPORAN 2.h5")

face_model = load_face_model()
digit_model = load_digit_model()

# =============================
# TABS UNTUK DUA MODEL
# =============================
tab1, tab2 = st.tabs(["😊 Ekspresi Wajah", "🔢 Klasifikasi Digit"])

# =============================
# TAB 1 - EKSPRESI WAJAH
# =============================
with tab1:
    st.subheader("😊 Deteksi Ekspresi Wajah Menggunakan Model PyTorch (.pt)")
    uploaded_file_face = st.file_uploader("📸 Upload Gambar Wajah", type=["jpg", "jpeg", "png"], key="face")

    if uploaded_file_face is not None:
        image_face = Image.open(uploaded_file_face).convert("L").resize((48, 48))
        st.image(image_face, caption="Gambar yang Diunggah", width=300)

        img_array = np.array(image_face) / 255.0
        tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).float()

        outputs = face_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()[0]
        pred_class = np.argmax(probs)

        # Label emosi
        classes = ["Jijik", "Senang", "Marah", "Sedih", "Takut"]
        st.success(f"🧠 Ekspresi Terdeteksi: **{classes[pred_class]}**")

        # Grafik probabilitas
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(classes, probs, color="#6a0dad")
        ax.set_ylabel("Probabilitas")
        ax.set_title("Distribusi Prediksi Emosi")
        st.pyplot(fig)
    else:
        st.info("Silakan upload gambar wajah terlebih dahulu.")

# =============================
# TAB 2 - KLASIFIKASI DIGIT
# =============================
with tab2:
    st.subheader("🔢 Klasifikasi Angka (0–9) Menggunakan Model TensorFlow (.h5)")
    uploaded_file_digit = st.file_uploader("📤 Upload Gambar Digit", type=["jpg", "jpeg", "png"], key="digit")

    if uploaded_file_digit is not None:
        image_digit = Image.open(uploaded_file_digit).convert("L").resize((28, 28))
        st.image(image_digit, caption="Gambar yang Diunggah", width=200)

        img_array = np.array(image_digit).reshape(1, 28, 28, 1) / 255.0
        prediction = digit_model.predict(img_array)
        probs = prediction[0]
        pred_digit = np.argmax(probs)

        st.success(f"🧮 Angka yang Terdeteksi: **{pred_digit}**")

        # Grafik probabilitas
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.bar(range(10), probs, color="#00b4d8")
        ax2.set_xlabel("Angka")
        ax2.set_ylabel("Probabilitas")
        ax2.set_title("Distribusi Prediksi Angka")
        st.pyplot(fig2)
    else:
        st.info("Silakan upload gambar digit terlebih dahulu.")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>✨ Dibuat oleh <b>Ine Lutfiatul Hanifah</b> | Statistika | Big Data Project 2025</p>", unsafe_allow_html=True)
