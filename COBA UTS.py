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
    .title { text-align: center; font-size: 32px; color: #4B7BE5; font-weight: bold; }
    .subheader { color: #333; font-size: 20px; text-align: center; margin-top: -10px; }
    .result-box { background-color: #F5F7FF; padding: 20px; border-radius: 15px; text-align: center;
                  box-shadow: 0px 2px 10px rgba(0,0,0,0.1); }
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

face_model, digit_model = load_models()

# ==========================
# UI HEADER & SIDEBAR CONTROLS
# ==========================
st.markdown("<div class='title'>🧠 Dashboard Klasifikasi Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Proyek UAS – Big Data & AI</div>", unsafe_allow_html=True)
st.write("")

# sidebar controls for digit preprocessing / debugging
st.sidebar.header("⚙️ Pengaturan")
logo_path = "LOGO USK.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
else:
    st.sidebar.warning("⚠️ Logo tidak ditemukan")

menu = st.sidebar.radio("Pilih Jenis Klasifikasi:", ["Ekspresi Wajah", "Digit Angka"])
st.sidebar.markdown("---")
st.sidebar.write("Pengaturan Digit (opsional):")
label_offset = st.sidebar.selectbox("Label offset (kalau model melabeli 1..10)", options=[0, -1])
show_debug = st.sidebar.checkbox("Tampilkan debug prediction vector", value=False)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# MAIN LOGIC
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar Input", use_container_width=True)

    # ----------------------
    # 1) Ekspresi Wajah (.pt) - tetap seperti sebelumnya
    # ----------------------
    if menu == "Ekspresi Wajah":
        st.subheader("🔍 Hasil Deteksi Ekspresi Wajah")
        try:
            results = face_model(img)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="📸 Deteksi Ekspresi", use_container_width=True)

            if len(results[0].boxes) == 0:
                st.warning("⚠️ Tidak ada wajah terdeteksi. Unggah gambar wajah close-up.")
            else:
                # gunakan nama kelas asli dari model YOLO
                model_labels = results[0].names  # dict {0:'label0',1:'label1',...}
                emoji_map = {
                    "senang": "😄", "bahagia": "😊", "sedih": "😢",
                    "marah": "😡", "takut": "😱", "jijik": "🤢",
                }
                for box in results[0].boxes:
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    label = model_labels.get(cls, "Tidak Dikenal").lower()
                    emoji = emoji_map.get(label, "🙂")
                    st.markdown(f"""
                        <div class='result-box'>
                            <h3>{emoji} Ekspresi: <b>{label.capitalize()}</b></h3>
                            <p style="font-size:16px;">🎯 Keyakinan: <b>{conf*100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat deteksi ekspresi: {e}")

    # ----------------------
    # 2) Digit Angka (.h5) - PERBAIKAN UTAMA
    # ----------------------
    elif menu == "Digit Angka":
        st.subheader("🔢 Hasil Klasifikasi Digit Angka")
        try:
            # ambil input shape model dan sesuaikan preprocessing
            input_shape = digit_model.input_shape  # e.g. (None, H, W, C)
            if len(input_shape) != 4:
                st.warning(f"Struktur input model tidak biasa: {input_shape}. Menggunakan (28,28,1) fallback.")
                target_size = (28, 28)
                channels = 1
            else:
                target_size = (input_shape[1], input_shape[2])
                channels = input_shape[3] if input_shape[3] is not None else 1

            # siapkan gambar sesuai channel
            if channels == 1:
                proc = img.convert("L")  # grayscale
            else:
                proc = img.convert("RGB")

            # resize sesuai model
            proc = proc.resize(target_size)
            # ubah ke array
            arr = image.img_to_array(proc).astype("float32")
            # normalisasi (default)
            arr /= 255.0

            # jika model menginginkan channel terakhir =1 tapi array shape (H,W,3), handle:
            if arr.ndim == 3 and arr.shape[2] != channels:
                if channels == 1:
                    # convert RGB->L then recreate array
                    proc2 = proc.convert("L")
                    arr = image.img_to_array(proc2).astype("float32") / 255.0
                else:
                    # convert grayscale->RGB
                    proc2 = proc.convert("RGB")
                    arr = image.img_to_array(proc2).astype("float32") / 255.0

            # tambahkan batch dim
            img_array = np.expand_dims(arr, axis=0)  # shape (1,H,W,C)

            # debug: tampilkan shapes jika diminta
            if show_debug:
                st.info(f"Model input_shape = {input_shape}  → menggunakan target_size={target_size}, channels={channels}")
                with st.expander("Debug: input array info"):
                    st.write("input array shape:", img_array.shape)
                    st.write("input array min/max:", float(img_array.min()), float(img_array.max()))

            # prediksi
            pred = digit_model.predict(img_array)
            # tampilkan prediction vector bila diminta
            if show_debug:
                with st.expander("Debug: prediction vector"):
                    st.write(pred)

            pred_label = int(np.argmax(pred[0]))
            prob = float(np.max(pred[0]))

            # apply label offset (jika model melabeli 1..10)
            if label_offset == -1:
                pred_label = pred_label - 1

            # safe-guard: pastikan 0..9
            pred_label = int(pred_label) % 10

            # tampilkan hasil
            colA, colB = st.columns(2)
            with colA:
                st.image(proc, caption="🖼️ Gambar Uji (preprocessed)", use_column_width=True)
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
            st.error("❌ Terjadi kesalahan saat klasifikasi digit:")
            st.error(e)

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk melakukan deteksi atau klasifikasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 – Dibuat oleh Ine Lutfia • Proyek UAS Big Data</p>", unsafe_allow_html=True)
