# app.py — BrailleBridge
# Streamlit app: camera/upload → YOLO → Braille → English + TTS + AI context

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from gtts import gTTS
from ultralytics import YOLO

from agents import chat_assistant, get_context
from braille_map import BRAILLE_MAP, dots_to_char

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #FEFCE8;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #111111 !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #F97316 !important;
        border-color: #F97316 !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #ea580c !important;
        border-color: #ea580c !important;
    }
    div[data-testid="stDownloadButton"] > button {
        background-color: #F97316 !important;
        border-color: #F97316 !important;
        color: #ffffff !important;
    }
</style>
"""

CHAR_TO_DOTS = {char: list(dots) for dots, char in BRAILLE_MAP.items()}


@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model(weights_path: str):
    return YOLO(weights_path)


def _read_image_bgr(source) -> np.ndarray | None:
    """Decode Streamlit camera/upload bytes to BGR numpy array."""
    if source is None:
        return None
    raw = source.getvalue() if hasattr(source, "getvalue") else source
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def _box_center_xy(box) -> tuple[float, float]:
    xyxy = box.xyxy[0].cpu().numpy()
    return float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2)


def _box_width(box) -> float:
    xyxy = box.xyxy[0].cpu().numpy()
    return float(xyxy[2] - xyxy[0])


def _label_for_box(model, box) -> str:
    cls_id = int(box.cls[0])
    return str(model.names.get(cls_id, model.names[cls_id])).strip()


def _char_from_class_label(label: str) -> str | None:
    """If YOLO class is A–Z, map through braille dot pattern → dots_to_char."""
    key = label.upper()
    if len(key) == 1 and key.isalpha():
        dots = CHAR_TO_DOTS.get(key.lower())
        if dots is not None:
            return dots_to_char(dots)
    if label.isdigit() and 1 <= int(label) <= 6:
        return dots_to_char([int(label)])
    return None


def _infer_dot_in_cell(cx: float, cy: float, xmin: float, ymin: float, w: float, h: float) -> int:
    nx = (cx - xmin) / max(w, 1.0)
    ny = (cy - ymin) / max(h, 1.0)
    col = 0 if nx < 0.5 else 1
    row = min(2, int(ny * 3.0))
    grid = {(0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 4, (1, 1): 5, (1, 2): 6}
    return grid[(col, row)]


def detections_to_text(model, result) -> str:
    """
    Group YOLO boxes into Braille cells (left-to-right), infer active dots per cell,
    and convert with dots_to_char.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ""

    entries = []
    for box in boxes:
        cx, cy = _box_center_xy(box)
        label = _label_for_box(model, box)
        conf = float(box.conf[0])
        entries.append(
            {
                "cx": cx,
                "cy": cy,
                "w": _box_width(box),
                "label": label,
                "conf": conf,
                "box": box,
            }
        )

    entries.sort(key=lambda e: e["cx"])
    widths = [e["w"] for e in entries if e["w"] > 0]
    median_w = float(np.median(widths)) if widths else 20.0
    gap = max(median_w * 0.8, 8.0)

    cells: list[list[dict]] = []
    current: list[dict] = []
    for entry in entries:
        if not current:
            current.append(entry)
            continue
        if entry["cx"] - current[-1]["cx"] > gap:
            cells.append(current)
            current = [entry]
        else:
            current.append(entry)
    if current:
        cells.append(current)

    chars: list[str] = []
    for cell_boxes in cells:
        if len(cell_boxes) == 1:
            from_class = _char_from_class_label(cell_boxes[0]["label"])
            if from_class is not None and cell_boxes[0]["conf"] >= 0.35:
                chars.append(from_class)
                continue

        xs = [b["cx"] for b in cell_boxes]
        ys = [b["cy"] for b in cell_boxes]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad = max(median_w * 0.25, 4.0)
        cell_w = max(xmax - xmin, median_w) + pad * 2
        cell_h = max(ymax - ymin, median_w) + pad * 2
        cell_xmin = xmin - pad
        cell_ymin = ymin - pad

        active_dots: set[int] = set()
        for item in cell_boxes:
            dot_label = item["label"]
            if dot_label.isdigit() and 1 <= int(dot_label) <= 6:
                active_dots.add(int(dot_label))
            else:
                active_dots.add(
                    _infer_dot_in_cell(
                        item["cx"], item["cy"], cell_xmin, cell_ymin, cell_w, cell_h
                    )
                )

        chars.append(dots_to_char(sorted(active_dots)))

    return "".join(chars).strip()


def run_inference(model, image_bgr: np.ndarray):
    return model(image_bgr, verbose=False)


def text_to_speech_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read()


def init_session_state():
    defaults = {
        "chat_history": [],
        "translated_text": "",
        "last_context": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    st.set_page_config(
        page_title="BrailleBridge",
        page_icon="👁",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    st.title("BrailleBridge")
    st.caption("Point. Read. Understand.")

    model = None
    if not MODEL_PATH.is_file():
        st.warning(
            f"Model weights not found at `{MODEL_PATH.name}`. "
            "Download **best.pt** from the link in `model/model_info.md` "
            "and place it in the project root."
        )
    else:
        try:
            model = load_model(str(MODEL_PATH))
        except Exception as exc:
            st.warning(f"Could not load model: {exc}")

    col_cam, col_upload = st.columns(2)
    with col_cam:
        camera_photo = st.camera_input("Capture Braille with your camera")
    with col_upload:
        uploaded = st.file_uploader(
            "Or upload an image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )

    image_source = camera_photo or uploaded
    if image_source is None:
        st.warning("Take a photo or upload an image to translate Braille.")
        _render_context_and_chat()
        return

    if model is None:
        st.warning("Translation is unavailable until **best.pt** is installed.")
        _render_context_and_chat()
        return

    image_bgr = _read_image_bgr(image_source)
    if image_bgr is None:
        st.warning("Could not read the image. Try another photo or file format.")
        _render_context_and_chat()
        return

    with st.spinner("Running Braille detection…"):
        results = run_inference(model, image_bgr)

    if not results:
        st.warning("No inference result returned.")
        _render_context_and_chat()
        return

    result = results[0]
    translated = detections_to_text(model, result)

    left, right = st.columns([1, 1])
    with left:
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption="Input image",
            use_container_width=True,
        )
    with right:
        plotted = result.plot()
        st.image(
            cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB),
            caption="YOLO detections",
            use_container_width=True,
        )

    if not translated:
        st.warning(
            "No Braille characters detected. Try better lighting, closer framing, "
            "or a clearer view of the dots."
        )
        st.session_state.translated_text = ""
        _render_context_and_chat()
        return

    st.session_state.translated_text = translated
    st.success(f"**Translation:** {translated}")

    try:
        audio_bytes = text_to_speech_bytes(translated)
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception as exc:
        st.warning(f"Text-to-speech failed: {exc}")

    st.download_button(
        label="Download transcript (.txt)",
        data=translated,
        file_name="braillebridge_transcript.txt",
        mime="text/plain",
    )

    _render_context_and_chat()


def _render_context_and_chat():
    st.divider()
    st.subheader("Context")
    translated = st.session_state.get("translated_text", "")
    if translated:
        with st.spinner("Generating context…"):
            context = get_context(translated)
        st.session_state.last_context = context
        st.info(context)
    else:
        st.caption("Context appears here after a successful translation.")

    st.divider()
    st.subheader("Ask BrailleBridge")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about this translation…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        prior = st.session_state.chat_history[:-1]
        with st.spinner("Thinking…"):
            reply = chat_assistant(
                prompt,
                st.session_state.get("translated_text", ""),
                prior,
            )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()
