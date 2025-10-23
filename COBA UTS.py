# dashboard_uts_final.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import time

# models (may be heavy) — load lazily with cache_resource
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ==========================
# Page config & styling
# ==========================
st.set_page_config(page_title="INÉ VISION STATION - UTS", page_icon="🤖", layout="wide")

st.markdown("""
<style>
/* ---------- general ---------- */
body {
  background: radial-gradient(circle at top left, #071028, #0f2740 60%, #142b3b 100%);
  color: #E6F7FF;
  font-family: "Poppins", sans-serif;
}
.header-title {
  text-align:center;
  font-size:40px;
  font-weight:800;
  color:#7EE7FF;
  text-shadow:0 0 18px rgba(126,231,255,0.25);
  margin-bottom:2px;
}
.header-sub {
  text-align:center;
  color:#BEEAF6;
  margin-top: -6px;
  margin-bottom: 18px;
}
/* glass card */
.card {
  background: rgba(255,255,255,0.03);
  border-radius:18px;
  padding:18px;
  box-shadow: 0 6px 30px rgba(0,200,255,0.06);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(126,231,255,0.06);
}
/* neon */
.neon {
  color:#9FF6FF;
  text-shadow:0 0 12px rgba(0,255,255,0.12), 0 0 22px rgba(0,200,255,0.06);
  font-weight:700;
}
.small-muted { color:#9ACFD9; font-size:14px; }
.footer { text-align:center; color:#7FBEDC; margin-top:30px; font-size:13px; }
/* buttons */
.big-btn {
  background: linear-gradient(90deg,#00C9FF,#0072FF);
  color:white; padding:10px 20px; border-radius:20px; font-weight:700;
  border:none; cursor:pointer;
}
.big-btn:hover { transform: scale(1.02); }
/* typing box */
.typing {
  border-left: 3px solid rgba(126,231,255,0.6);
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(255,255,255,0.02);
  color:#DFF8FF;
  font-style: italic;
}
/* layout helpers */
.center { text-align:center }
</style>
""", unsafe_allow_html=True)

# ==========================
# Session State: page navigation + small history
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "history" not in st.session_state:
    st.session_state.history = []

def go(page):
    st.session_state.page = page

# ==========================
# Load models (cached)
# ==========================
@st.cache_resource
def load_models():
    # paths (adjust names if berbeda)
    face_path = "model/Ine Lutfiatul Hanifah_Laporan 4 Bigdata.pt"
    digit_path = "model/INELUTFIATULHANIFAH_LAPORAN 2.h5"

    face_model = None
    digit_model = None

    if os.path.exists(face_path):
        try:
            face_model = YOLO(face_path)
        except Exception as e:
            face_model = None
    if os.path.exists(digit_path):
        try:
            digit_model = tf.keras.models.load_model(digit_path)
        except Exception as e:
            digit_model = None

    return face_model, digit_model

face_model, digit_model = load_models()

# helpful small "AI typing" function (lightweight)
def ai_typing(message, rapid=0.03):
    """Show a 'typing' effect by progressively writing message in a placeholder."""
    place = st.empty()
    text = ""
    for ch in message:
        text += ch
        place.markdown(f"<div class='typing'>{text}</div>", unsafe_allow_html=True)
        time.sleep(rapid)
    # leave for a short while
    time.sleep(0.08)
    return

# ==========================
# NAV BAR (sidebar)
# ==========================
st.sidebar.markdown("## 🔭 Navigasi Slide")
page = st.sidebar.radio("", ["Home (Cover)", "Deteksi Ekspresi", "Klasifikasi Digit", "Tentang"], index=["home","ekspresi","angka","tentang"].index(st.session_state.page) if st.session_state.page in ["home","ekspresi","angka","tentang"] else 0)
# map radio selection to internal page
page_map = {
    "Home (Cover)": "home",
    "Deteksi Ekspresi": "ekspresi",
    "Klasifikasi Digit": "angka",
    "Tentang": "tentang"
}
if page_map.get(page, "home") != st.session_state.page:
    st.session_state.page = page_map.get(page, "home")

st.sidebar.markdown("---")
st.sidebar.markdown("⚙️ Tips: Gunakan gambar close-up wajah untuk hasil ekspresi optimal.")
if os.path.exists("LOGO USK.png"):
    st.sidebar.image("LOGO USK.png", width=140)
else:
    st.sidebar.caption("Logo tidak ditemukan (letakkan LOGO USK.png di folder projek)")

# ==========================
# PAGE: HOME / COVER
# ==========================
if st.session_state.page == "home":
    st.markdown("<div class='header-title'>INÉ VISION STATION</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>UTS Big Data & AI — Klasifikasi Ekspresi Wajah & Digit Angka</div>", unsafe_allow_html=True)
    st.markdown("<div class='center'><div class='card' style='display:inline-block; max-width:900px'>", unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("<h3 class='neon'>Presentasi Interaktif</h3>", unsafe_allow_html=True)
        st.markdown("<p class='small-muted'>Slide-style dashboard: Cover → Ekspresi → Digit → Tentang</p>", unsafe_allow_html=True)
        st.markdown("""
            <ul style='color:#CFF6FF'>
                <li>Antarmuka futuristik, responsif, dan interaktif.</li>
                <li>Deteksi ekspresi memakai YOLOv8 (tampilkan 1 hasil terbaik).</li>
                <li>Klasifikasi digit: preprocessing otomatis (shape-safe) + visual feedback.</li>
            </ul>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("✨ MULAI EXPLORASI", key="start"):
            go("ekspresi")
    with col2:
        if os.path.exists("cover.png"):
            st.image("cover.png", width=240)
        else:
            # small animated-ish block (emoji)
            st.markdown("<div style='font-size:48px; text-align:center;'>🤖</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted center'>INÉ VISION</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>© 2025 – Ine Lutfia • Proyek UTS Big Data & AI</div>", unsafe_allow_html=True)

# ==========================
# PAGE: DETEKSI EKSPRESI (1 best detection)
# ==========================
elif st.session_state.page == "ekspresi":
    st.markdown("<div class='header-title'>Deteksi Ekspresi Wajah</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>Sistem akan menampilkan <b>satu</b> hasil deteksi terbaik (confidence tertinggi)</div>", unsafe_allow_html=True)

    if face_model is None:
        st.error("❌ Model YOLO untuk ekspresi tidak ditemukan atau gagal dimuat. Pastikan path model benar.")
        st.button("⬅️ Kembali ke Home", on_click=lambda: go("home"))
    else:
        uploaded = st.file_uploader("📤 Unggah gambar wajah (jpg/png)", type=["jpg","jpeg","png"])
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar input", use_column_width=True)

            # small typing intro
            ai_typing("Menganalisis... memilih kotak deteksi yang paling yakin.")

            try:
                results = face_model(img)  # YOLO results
                if len(results) == 0 or len(results[0].boxes) == 0:
                    st.warning("⚠️ Tidak ada wajah jelas terdeteksi. Coba gambar close-up.")
                else:
                    # choose best box by confidence
                    boxes = results[0].boxes
                    best_idx = int(np.argmax([float(b.conf[0]) for b in boxes]))
                    best_box = boxes[best_idx]
                    conf = float(best_box.conf[0]) if best_box.conf is not None else 0.0
                    cls = int(best_box.cls[0]) if best_box.cls is not None else 0
                    label = results[0].names.get(cls, "Tidak Dikenal").lower()

                    # show annotated image (YOLO plot)
                    annotated = results[0].plot()  # numpy array
                    st.image(annotated, caption=f"Annotated (best box conf={conf*100:.2f}%)", use_column_width=True)

                    # friendly AI response
                    emoji_map = {"senang":"😄","bahagia":"😊","sedih":"😢","marah":"😡","takut":"😱","jijik":"🤢"}
                    emoji = emoji_map.get(label, "🙂")
                    st.markdown(f"""
                        <div class='card'>
                            <h3 class='neon'>{emoji} {label.capitalize()}</h3>
                            <p>Akurasi deteksi: <b>{conf*100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                    # personality quick tip
                    if label in ["sedih", "takut"]:
                        st.info("💬 Aku lihat kamu terdeteksi sedih/takut — semoga harimu membaik! 😊")
                    elif label in ["bahagia", "senang"]:
                        st.success("💬 Senyummu keren! Terus semangat! ✨")
                    elif label == "marah":
                        st.warning("💬 Tenangkan diri dulu, tarik napas 🌬️")

            except Exception as e:
                st.error(f"❌ Terjadi error saat inferensi YOLO: {e}")

        # navigation
        col1, col2 = st.columns([1,1])
        with col1:
            st.button("⬅️ Kembali ke Home", on_click=lambda: go("home"))
        with col2:
            st.button("➡️ Lanjut ke Klasifikasi Digit", on_click=lambda: go("angka"))

# ==========================
# PAGE: KLASIFIKASI DIGIT (robust preprocessing + attractive)
# ==========================
elif st.session_state.page == "angka":
    st.markdown("<div class='header-title'>Klasifikasi Digit Angka</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>Preprocessing otomatis (resize, grayscale/RGB, reshape) — aman ke model</div>", unsafe_allow_html=True)

    # show model availability
    if digit_model is None:
        st.error("❌ Model Keras (.h5) untuk digit tidak ditemukan atau gagal dimuat. Pastikan path benar.")
        st.button("⬅️ Kembali ke Ekspresi", on_click=lambda: go("ekspresi"))
    else:
        uploaded = st.file_uploader("📥 Unggah gambar digit/tulisan tangan (png/jpg)", type=["png","jpg","jpeg"], key="digit_upload")
        colL, colR = st.columns([2,1])
        if uploaded is None:
            with colR:
                st.markdown("<div class='card'><p class='small-muted'>Tips: Untuk hasil terbaik, unggah gambar digit yang kontras (angka gelap di latar terang), crop sekitar angka.</p></div>", unsafe_allow_html=True)
        if uploaded is not None:
            # read image and display
            raw = Image.open(uploaded)
            st.image(raw, caption="Gambar Input (raw)", use_column_width=True)

            # ========== determine expected input shape ==========
            try:
                input_shape = digit_model.input_shape  # e.g. (None, H, W, C)
            except Exception:
                input_shape = None

            # fallback defaults
            H = 28; W = 28; C = 1
            if input_shape and len(input_shape) == 4:
                # input_shape may contain None in dims; handle safely
                _, h, w, c = input_shape
                H = int(h) if (h is not None and isinstance(h, (int, np.integer))) else H
                W = int(w) if (w is not None and isinstance(w, (int, np.integer))) else W
                C = int(c) if (c is not None and isinstance(c, (int, np.integer))) else C

            # preprocess: convert color properly
            try:
                if C == 1:
                    proc = ImageOps.grayscale(raw)
                else:
                    proc = raw.convert("RGB")
                proc = proc.resize((W, H))
                arr = image.img_to_array(proc).astype("float32") / 255.0

                # ensure arr shape (H,W,C)
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, -1)
                if arr.shape[-1] != C:
                    # try to convert channel to expected
                    if C == 1:
                        proc2 = ImageOps.grayscale(proc)
                        arr = image.img_to_array(proc2).astype("float32") / 255.0
                    else:
                        proc2 = proc.convert("RGB")
                        arr = image.img_to_array(proc2).astype("float32") / 255.0

                # final batch axis
                img_array = np.expand_dims(arr, axis=0)  # shape (1,H,W,C)

                # optional debug info
                if st.checkbox("Tampilkan debug preprocessing", value=False):
                    st.write("Model input_shape:", input_shape)
                    st.write("Prepared array shape:", img_array.shape)
                    st.write("Array min/max:", float(img_array.min()), float(img_array.max()))

                # show preprocessed preview (small)
                st.image(Image.fromarray((arr.squeeze()*255).astype('uint8')), caption="Preview Preprocessed", width=140)

                # inference
                with st.spinner("⏳ Model sedang memprediksi..."):
                    # safety: wrap predict in try
                    try:
                        pred = digit_model.predict(img_array)
                        pred = np.array(pred)  # ensure numpy
                        pred_label = int(np.argmax(pred[0]))
                        prob = float(np.max(pred[0]))
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi model: {e}")
                        pred_label, prob = None, None

                if pred_label is not None:
                    # friendly visuals based on confidence
                    color = "#7EE7FF" if prob >= 0.85 else ("#FFD166" if prob >= 0.6 else "#FF9AA2")
                    emoji = "🚀" if prob >= 0.9 else ("✨" if prob >= 0.75 else "🤔")

                    st.markdown(f"""
                        <div class='card'>
                            <h2 style='color:{color}; font-size:44px; margin:6px'>{emoji} Prediksi: <b>{pred_label}</b></h2>
                            <p class='small-muted'>Keyakinan model: <b>{prob*100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                    # parity message
                    if pred_label % 2 == 0:
                        st.success("✅ Angka ini genap — stabil & balanced")
                    else:
                        st.warning("⚡ Angka ini ganjil — cenderung dinamis")

                    # append to session history
                    st.session_state.history.append({"label": pred_label, "prob": prob, "time": time.strftime("%H:%M:%S")})

                    # mini stats & history
                    if st.checkbox("Tampilkan histori prediksi sesi ini"):
                        hist = st.session_state.history[-10:][::-1]  # last 10 reversed
                        for h in hist:
                            st.markdown(f"- `{h['time']}` → **{h['label']}** (conf {h['prob']*100:.1f}%)")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan preprocessing: {e}")

        # navigation buttons
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.button("⬅️ Kembali ke Ekspresi", on_click=lambda: go("ekspresi"))
        with col3:
            st.button("ℹ️ Tentang Proyek", on_click=lambda: go("tentang"))

# ==========================
# PAGE: TENTANG / PROYEK
# ==========================
elif st.session_state.page == "tentang":
    st.markdown("<div class='header-title'>Tentang Proyek</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
        <h3 class='neon'>Ringkasan</h3>
        <p class='small-muted'>
        Project ini menggabungkan dua kemampuan computer vision:
        YOLOv8 untuk deteksi ekspresi wajah (menampilkan 1 kotak terbaik), dan
        model Keras (CNN) untuk klasifikasi digit.
        </p>
        <p class='small-muted'>
        Tujuan: membuat antarmuka interaktif (slide-like) yang menarik untuk presentasi UTS.
        </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>")
    if st.button("🏠 Kembali ke Home"):
        go("home")

# Footer
st.markdown("<div class='footer'>© 2025 – Ine Lutfia • UTS Big Data & AI</div>", unsafe_allow_html=True)
