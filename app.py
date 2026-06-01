import streamlit as st
import cv2
import numpy as np
import io
from gtts import gTTS
from ultralytics import YOLO
from agents import get_context, chat_assistant
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BrailleBridge",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
# ── Brand CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 
* { font-family: 'Inter', sans-serif; }
 
.stApp { background-color: #FEFCE8 !important; }
 
/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1200px; }
 
/* ── Header ── */
.bb-header {
    background: #111111;
    padding: 14px 24px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}
.bb-header-left { display: flex; align-items: center; gap: 12px; }
.bb-icon {
    width: 36px; height: 36px;
    background: #F97316;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.bb-title { color: #fff; font-weight: 700; font-size: 17px; margin: 0; }
.bb-sub { color: #888; font-size: 11px; margin: 0; }
.bb-badge {
    background: #F97316; color: #111;
    font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
}
 
/* ── Labels ── */
.bb-label {
    font-size: 11px; color: #999; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 10px;
}
 
/* ── Cards ── */
.bb-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
 
/* ── Translation box ── */
.bb-translation {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 16px;
    margin: 12px 0;
}
.bb-translation-text {
    font-size: 24px; font-weight: 700;
    color: #111; letter-spacing: 0.1em; margin: 0;
}
.bb-translation-meta { font-size: 11px; color: #888; margin: 4px 0 0; }
 
/* ── Context ── */
.bb-context {
    border-left: 3px solid #F97316;
    background: #fff;
    border-top: 1px solid #e5e7eb;
    border-right: 1px solid #e5e7eb;
    border-bottom: 1px solid #e5e7eb;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin-bottom: 20px;
}
.bb-context-label {
    font-size: 11px; color: #F97316; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0 0 4px;
}
.bb-context-text { font-size: 13px; color: #555; margin: 0; }
 
/* ── Buttons ── */
.stButton > button {
    background: #F97316 !important;
    color: #111 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    width: 100% !important;
}
.stDownloadButton > button {
    background: #111 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
 
/* ── Chat ── */
[data-testid="stChatInput"] textarea {
    background: #fff !important;
    color: #111 !important;
    border: 1.5px solid #F97316 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] {
    background: #fff !important;
}
.stChatFloatingInputContainer {
    background: #FEFCE8 !important;
    border-top: 1px solid #e5e7eb !important;
}
 
/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #F97316 !important;
    border-radius: 10px !important;
    background: #fff !important;
}
 
/* ── Radio ── */
.stRadio [data-baseweb="radio"] { gap: 16px; }
section[data-testid="stBottom"] {
    background-color: #FEFCE8 !important;
}
section[data-testid="stBottom"] > div {
    background-color: #FEFCE8 !important;
}
</style>
""", unsafe_allow_html=True)
 
# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bb-header">
    <div class="bb-header-left">
        <div class="bb-icon">👁</div>
        <div>
            <p class="bb-title">BrailleBridge</p>
            <p class="bb-sub">Point. Read. Understand.</p>
        </div>
    </div>
    <span class="bb-badge">AI-powered</span>
</div>
""", unsafe_allow_html=True)
 
# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
 
# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        return None
 
model = load_model()
 
if model is None:
    st.error("⚠️ best.pt not found. Place your trained model in the project root.")
    st.stop()
 
# ── Helper functions ──────────────────────────────────────────────────────────
def run_translation(image_bgr):
    results = model(image_bgr, verbose=False)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x_center = float(box.xywh[0][0])
        y_center = float(box.xywh[0][1])
        height = float(box.xywh[0][3])
        if conf > 0.4:
            letter = model.names[cls_id]
            detections.append((x_center, y_center, height, letter, conf))

    translated_text = ""
    avg_conf = 0

    if detections:
        avg_height = sum(d[2] for d in detections) / len(detections)
        row_threshold = avg_height * 0.6

        detections.sort(key=lambda d: d[1])

        rows = []
        current_row = [detections[0]]
        for det in detections[1:]:
            if abs(det[1] - current_row[0][1]) <= row_threshold:
                current_row.append(det)
            else:
                rows.append(current_row)
                current_row = [det]
        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda d: d[0])

        row_texts = [" ".join(d[3] for d in row) for row in rows]
        translated_text = "  ".join(row_texts).upper()
        avg_conf = sum(d[4] for d in detections) / len(detections)
    annotated = results[0].plot()
    return translated_text, len(detections), avg_conf, annotated
 
def make_audio(text):
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None
 
# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")
 
# ── LEFT: Input ───────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<p class="bb-label">Input</p>', unsafe_allow_html=True)
 
    mode = st.radio(
        "Input mode",
        ["Upload Image", "Camera"],
        horizontal=True,
        label_visibility="collapsed"
    )
 
    image_bgr = None
 
    if mode == "Upload Image":
        uploaded = st.file_uploader(
            "Drop a Braille image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible"
        )
        if uploaded:
            arr = np.frombuffer(uploaded.read(), np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        cam = st.camera_input("Take a photo of Braille")
        if cam:
            arr = np.frombuffer(cam.read(), np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
 
    translate_btn = st.button("👁 Translate Braille", type="primary")
 
# ── RIGHT: Output ─────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<p class="bb-label">Translation</p>', unsafe_allow_html=True)
 
    if translate_btn and image_bgr is not None:
        with st.spinner("Detecting Braille dots..."):
            text, count, conf, annotated = run_translation(image_bgr)
            st.session_state.translated_text = text
            st.session_state.last_result = {
                "input": image_bgr,
                "annotated": annotated,
                "text": text,
                "count": count,
                "conf": conf
            }
 
    if st.session_state.last_result:
        r = st.session_state.last_result
 
        # Images
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(
                cv2.cvtColor(r["input"], cv2.COLOR_BGR2RGB),
                caption="Input image"
            )
        with img_col2:
            st.image(
                cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB),
                caption="YOLO detections"
            )
 
        if r["text"]:
            # Translation result
            st.markdown(f"""
            <div class="bb-translation">
                <p class="bb-translation-text">{r["text"]}</p>
                <p class="bb-translation-meta">
                    {r["count"]} characters detected &nbsp;·&nbsp;
                    Avg confidence: {r["conf"]:.0%}
                </p>
            </div>
            """, unsafe_allow_html=True)
 
            # Audio
            audio = make_audio(r["text"])
            if audio:
                st.audio(audio, format="audio/mp3")
 
            # Download
            st.download_button(
                "📄 Download Transcript (.txt)",
                data=r["text"],
                file_name="braille_transcript.txt",
                mime="text/plain"
            )
        else:
            st.warning("No Braille characters detected. Try a clearer or closer image.")
 
    elif not translate_btn:
        st.info("Upload an image and press **Translate Braille** to begin.")
 
# ── Context + Chat ────────────────────────────────────────────────────────────
if st.session_state.translated_text:
    st.markdown("---")
 
    # Context
    context = get_context(st.session_state.translated_text)
    st.markdown(f"""
    <div class="bb-context">
        <p class="bb-context-label">✨ AI Context</p>
        <p class="bb-context-text">{context}</p>
    </div>
    """, unsafe_allow_html=True)
 
    # Chat
    st.markdown('<p class="bb-label">Ask BrailleBridge</p>', unsafe_allow_html=True)
 
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
 
    if user_q := st.chat_input("Ask about this translation..."):
        with st.chat_message("user"):
            st.write(user_q)
 
        reply, st.session_state.chat_history = chat_assistant(
            user_q,
            st.session_state.translated_text,
            st.session_state.chat_history
        )
 
        with st.chat_message("assistant"):
            st.write(reply)
