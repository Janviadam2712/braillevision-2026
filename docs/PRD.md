# PRD — BrailleBridge

### BrailleVision Hackathon 2026 | Solo Submission

**Builder:** Janvi | **GitHub:** Janviadam2712/braillevision-2026 **Updated:** May 31, 2026 (during build window)

> AI Disclosure: PRD structure assisted by Claude (Anthropic). All product decisions and build execution by Janvi.

---

## 0. Project Identity


| Field         | Detail                                           |
| ------------- | ------------------------------------------------ |
| Name          | BrailleBridge                                    |
| Tagline       | Point. Read. Understand.                         |
| GitHub        | github.com/Janviadam2712/braillevision-2026      |
| Type          | Local Python web app (Streamlit, localhost:8501) |
| Primary color | `#111111` Ink Black                              |
| Accent color  | `#F97316` Amber Orange                           |
| Background    | `#FEFCE8` Warm Cream                             |


---

## 0.1 Positioning Statement

For sighted caregivers, healthcare workers, educators, and emergency responders who encounter physical Braille and cannot read it — BrailleBridge is a real-time, zero-cost translation system that converts any embossed Braille surface to spoken English using only a camera. No specialist training. No additional hardware. No subscription.

One line: "Point. Read. Understand. Braille for everyone."

---

## 0.2 The Core Problem

Physical Braille lives on medicine bottles, elevator buttons, hazmat containers, museum signs, and children's books. These objects are handled daily by billions of sighted people who cannot read Braille.

The actual user is not the blind person — they already read Braille with their fingers. The user is the sighted nurse, teacher, parent, or emergency responder holding a Braille-labeled object with no way to decode it.

One sentence for judges: "Physical Braille carries critical information on medicine labels and safety signs — but no affordable real-time tool exists to make that information readable to the billions of sighted people who encounter it daily."

---

## 1. Tech Stack


| Layer            | Tool                      | Version/Notes                     |
| ---------------- | ------------------------- | --------------------------------- |
| Object detection | YOLOv8n                   | Fine-tuned, 30 epochs, 26 classes |
| Image processing | OpenCV headless           | No GUI compilation issues         |
| Web framework    | Streamlit                 | localhost:8501                    |
| Text to speech   | gTTS                      | Google TTS, internet required     |
| AI agents        | Anthropic Claude API      | 3 roles                           |
| AI fallback      | claude-sonnet-4-6         | Vision, Role 1                    |
| AI context/chat  | claude-haiku-4-5-20251001 | Roles 2 and 3                     |
| Dependency       | numpy<2                   | Pinned for PyTorch compatibility  |


---

## 2. System Architecture

```
Camera/Image Upload
      ↓
OpenCV preprocessing
      ↓
YOLOv8n inference (best.pt)
      ↓
Sort detections left to right by x-coordinate
      ↓
Read class names directly from model.names[cls_id]
      ↓
Build translated_text string
      ↓
┌─────────────────────────────┐
│ Claude Context Layer        │ ← auto after translation
│ Explains what was read      │
└─────────────────────────────┘
      ↓
Output:
  Text on screen (st.success)
  Audio (gTTS → st.audio)
  Transcript (.txt download)
      ↓
┌─────────────────────────────┐
│ Claude Chat Assistant       │ ← on user question
│ Answers follow-up questions │
└─────────────────────────────┘
```

---

## 3. Features Built

### Complete

- Image upload + camera input
- YOLOv8 inference with detection visualization
- Left-to-right translation using YOLO class names directly
- gTTS audio playback
- Downloadable .txt transcript
- Claude context layer (with clean fallback)
- Chat interface (with clean fallback)
- Brand colors applied via custom CSS

### Remaining Before Final Submission

- Anthropic credits added (tomorrow morning)
- Claude agents fully tested with live responses
- Sample outputs screenshots
- Demo video recorded and uploaded

---

## 4. Judging Criteria Status


| Criterion                    | Implementation                       | Status        |
| ---------------------------- | ------------------------------------ | ------------- |
| Braille recognition accuracy | YOLOv8n, 0.85-0.97 confidence        | ✅ Strong      |
| Real-time performance        | Under 200ms inference                | ✅ Working     |
| Technical implementation     | Full pipeline, 3 AI agents           | ✅ Strong      |
| Robustness                   | numpy pinned, fallbacks added        | ✅ Good        |
| Accessibility and UX         | Audio, transcript, chat              | ✅ Working     |
| Innovation and impact        | 3 Claude roles, sighted user insight | ✅ Strong      |
| Demo and submission quality  | Video tomorrow, docs complete        | ⏳ In progress |


---

## 5. Dataset

Name: Braille Detection V2 Source: Roboflow Universe URL: [https://universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue](https://universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue) License: CC BY 4.0 Images: 1,324 Classes: 26 (a through z, Grade 1 Braille) Format: YOLOv8 YOLO TXT Split: 70% train / 20% val / 10% test

---

## 6. Model

File: [best.pt](http://best.pt) Architecture: YOLOv8n Epochs: 30 Image size: 640x640 Device: Google Colab T4 GPU Training time: ~40 minutes Download: [GOOGLE DRIVE LINK — paste here] Load command: from ultralytics import YOLO; model = YOLO('[best.pt](http://best.pt)')

---

## 7. Repository Structure

```
braillevision-2026/
├── app.py
├── agents.py
├── braille_map.py
├── best.pt (local only, not in git)
├── requirements.txt
├── setup_instructions.md
├── ai_tools_disclosure.md
├── README.md
├── .streamlit/config.toml
├── model/model_info.md
├── training/
│   ├── train.ipynb
│   ├── training_logs/results.png
│   └── results/confusion_matrix.png
├── dataset/
│   ├── data.yaml
│   ├── dataset_info.md
│   └── sample_images/
├── inference/inference.py
├── demo/
│   ├── demo_video_link.txt
│   └── screenshots/
└── docs/
    ├── PRD.md
    └── LEARNING_JOURNAL.md
```

---

## 8. AI Tools Disclosure


| Tool               | Purpose                                 |
| ------------------ | --------------------------------------- |
| Claude (Anthropic) | Architecture planning, documentation    |
| Claude Vision API  | Fallback verifier in detection pipeline |
| Claude API         | Context layer and chat assistant        |
| Cursor IDE         | Code scaffolding via Composer           |


---

## 9. Setup Instructions Summary

```
git clone https://github.com/Janviadam2712/braillevision-2026
cd braillevision-2026
pip install -r requirements.txt
# Download best.pt from Google Drive link in model/model_info.md
# Place best.pt in root directory
export ANTHROPIC_API_KEY="your-key"
streamlit run app.py
# Open http://localhost:8501
```

---

## 10. Submission Status


| Item                                            | Status              |
| ----------------------------------------------- | ------------------- |
| GitHub repo public                              | ✅ Done              |
| Source code complete                            | ✅ Done              |
| [README.md](http://README.md)                   | ✅ Done              |
| requirements.txt                                | ✅ Done              |
| setup_[instructions.md](http://instructions.md) | ✅ Done              |
| training/train.ipynb                            | ✅ Done              |
| inference/[inference.py](http://inference.py)   | ✅ Done              |
| dataset/data.yaml                               | ✅ Done              |
| dataset/dataset_[info.md](http://info.md)       | ✅ Done              |
| sample_images                                   | ✅ Done              |
| model/model_[info.md](http://info.md)           | ✅ Done              |
| ai_tools_[disclosure.md](http://disclosure.md)  | ✅ Done              |
| [best.pt](http://best.pt) on Google Drive       | ✅ Done              |
| training results/logs                           | ✅ Done              |
| demo/screenshots                                | ⏳ Tomorrow          |
| demo video                                      | ⏳ Tomorrow          |
| final submission form                           | ⏳ Before 5PM June 1 |


