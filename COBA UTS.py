import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# ==========================
# Utilities
# ==========================
def pil_to_rgb_array(pil_img):
    """Convert PIL Image to RGB numpy array (H,W,3)"""
    return np.array(pil_img.convert("RGB"))

def ensure_rgb_and_resize(pil_img, size=(224,224)):
    pil_rgb = pil_img.convert("RGB")
    pil_resized = pil_rgb.resize(size)
    arr = image.img_to_array(pil_resized).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0), pil_rgb  # return batch array and rgb PIL

# ==========================
# Load Models (cached)
# ==========================
# Fallback: jika st.cache_resource tidak tersedia, pakai st.cache
cache_loader = getattr(st, "cache_resource", None) or getattr(st, "cache", None)

@cache_loader
def load_models():
    # Pastikan path benar. Ganti nama file jika perlu.
    yolo_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    clf_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    if not os.path.exists(yolo_path):
        st.error(f"YOLO model tidak ditemukan di: {yolo_path}")
    if not os.path.exists(clf_path):
        st.error(f"Classifier .h5 tidak ditemukan di: {clf_path}")

    yolo_model = YOLO(yolo_path)  # bisa memerlukan GPU/CPU konfig
    classifier = tf.keras.models.load_model(clf_path)
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")
menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membuka gambar: {e}")
        st.stop()

    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        try:
            # Pastikan input ke YOLO berupa numpy array RGB
            np_img = pil_to_rgb_array(img)
            results = yolo_model(np_img)  # memanggil model
            # plot hasil (bisa jadi numpy array BGR atau RGB tergantung versi)
            plotted = results[0].plot()  # biasanya numpy array
            # Jika hasil adalah BGR, konversi ke RGB (cek dengan asumsi OpenCV)
            if plotted is not None and isinstance(plotted, np.ndarray):
                # jika shape (H,W,3) dan dtype uint8 -> convert BGR->RGB apabila perlu
                # Cek apakah warna tampak "aneh" tidak bisa otomatis, tapi umum perlu cvtColor:
                plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                st.image(plotted_rgb, caption="Hasil Deteksi", use_container_width=True)
            else:
                st.image(plotted, caption="Hasil Deteksi", use_container_width=True)
        except Exception as e:
            st.error(f"Error saat deteksi objek: {e}")

    elif menu == "Klasifikasi Gambar":
        try:
            # Pastikan RGB + ukuran sesuai
            img_array, pil_rgb = ensure_rgb_and_resize(img, size=(224,224))
            preds = classifier.predict(img_array)
            class_index = int(np.argmax(preds, axis=1)[0])
            prob = float(np.max(preds))
            st.write("### Hasil Prediksi:", class_index)
            st.write("Probabilitas:", round(prob, 4))
            # Jika punya mapping label -> tampilkan nama kelas
            # labels = ["kelas0", "kelas1", ...]
            # st.write("Kelas:", labels[class_index])
        except Exception as e:
            st.error(f"Error saat klasifikasi: {e}")
if menu == "Deteksi Objek (YOLO)":
    # Pastikan gambar dalam format RGB
    np_img = np.array(img.convert("RGB"))

    # Jalankan YOLO
    results = yolo_model(np_img)
    result = results[0]  # ambil hasil pertama

    # Plot hasil deteksi (gambar dengan box)
    plotted = result.plot()  # hasil default BGR
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    # Tampilkan gambar hasil deteksi
    st.image(plotted_rgb, caption="Hasil Deteksi", use_container_width=True)

    # ==============================
    # TAMBAHKAN CAPTION DI BAWAHNYA
    # ==============================
    if result.boxes is not None and len(result.boxes) > 0:
        # Ambil label dari setiap box
        detected_labels = [result.names[int(c)] for c in result.boxes.cls]
        # Hitung berapa kali tiap label muncul
        label_counts = {label: detected_labels.count(label) for label in set(detected_labels)}

        # Tampilkan teks di bawah gambar
        st.markdown("### 📦 Objek Terdeteksi:")
        for label, count in label_counts.items():
            st.write(f"- **{label}** : {count} kali terdeteksi")
    else:
        st.info("⚠️ Tidak ada objek terdeteksi pada gambar ini.")
