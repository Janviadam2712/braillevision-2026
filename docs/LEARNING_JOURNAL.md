# Learning Journal

# BrailleBridge — Learning Journal

**Builder:** Janvi | **Hackathon:** BrailleVision 2026 **Started:** May 30, 2026 | **Build Window:** May 31 4PM → June 1 5PM IST

> AI disclosure: Technical explanations and structure assisted by Claude (Anthropic). Build decisions, debugging, code implementation, and personal notes are Janvi's own work.

---

## Why This Project Exists

Physical Braille lives on medicine bottles, elevator buttons, hazmat labels, and safety signs. The sighted nurses, teachers, parents, and emergency responders who hold these objects every day cannot read them.

BrailleBridge solves this. Point any camera at embossed Braille → get English text + spoken audio + downloadable transcript + AI context.

Primary user: not the blind person — the sighted person in their world. Scale: not 253 million blind people — billions of sighted people who interact with a Braille-labeled world they cannot read.

One sentence: "Physical Braille carries critical information but no affordable real-time tool makes it readable to the billions of sighted people who encounter it daily."

---

## What I Learned

### Braille Structure

Six dot positions in a 2x3 grid per character:

```
Dot 1  Dot 4
Dot 2  Dot 5
Dot 3  Dot 6
```

64 possible combinations. 26 letters cover Grade 1 Braille (MVP scope). Studied from: pharmabraille.com/braille-alphabet

### What YOLOv8 Does

Object detection model fine-tuned to recognize Braille characters directly. Returns class name (the letter), bounding box coordinates, and confidence score. Confidence 0.85-0.97 achieved on real Braille test images. Key insight: YOLO outputs the letter name directly — no dot mapping needed. Sort detections by x-coordinate for correct left-to-right reading order.

### What [best.pt](http://best.pt) Is

Binary PyTorch model weights file. Cannot be opened like a document. Load only via: model = YOLO('[best.pt](http://best.pt)') Size: 5-7MB for yolov8n. Trained on T4 GPU, 30 epochs, 40 minutes.

### OpenCV Role

Preprocesses camera images before YOLO inference. Using opencv-python-headless (not opencv-python) to avoid Mac compilation issues.

### gTTS

Google Text-to-Speech. Converts translated text to audio bytes. Plays in browser via [st.audio](http://st.audio)(). Requires internet connection.

### Anthropic API — Three Agent Roles

Role 1 (fallback_verify): YOLO confidence below 0.6 triggers Claude Vision Role 2 (get_context): After translation, Claude explains what was read Role 3 (chat_assistant): User asks follow-up questions, Claude answers All wrapped in try/except so app never crashes without credits.

### Streamlit

Python web framework. Runs at localhost:8501. Start with: streamlit run [app.py](http://app.py) Never run with: python3 [app.py](http://app.py)

---

## Decisions Made


| Decision                                             | Why                                                             |
| ---------------------------------------------------- | --------------------------------------------------------------- |
| YOLOv8 reads class names directly                    | YOLO already knows the letter — dot mapping caused wrong output |
| opencv-python-headless                               | Regular opencv caused stuck compilation on Mac                  |
| numpy<2 pinned in requirements                       | Prevents PyTorch numpy conflict on judge machines               |
| ultralytics version pinned                           | Prevents version mismatch when judges install                   |
| Try/except on all Claude calls                       | App stays clean without API credits                             |
| Keep repo name braillevision-2026                    | Renaming mid-hackathon risks broken judge links                 |
| [best.pt](http://best.pt) on Google Drive not GitHub | File too large for git, causes HTTP 400 errors                  |
| Sighted caregiver as primary user                    | Blind users already read Braille — sighted users cannot         |
| Grade 1 Braille only                                 | Grade 2 contractions too complex for 25hr build                 |
| Jitter + Runway + CapCut for video                   | Jitter exports both Lottie and MP4, Runway for B-roll           |


---

## Bugs Hit and Fixed


| Bug                                  | What Happened                             | How Fixed                                    |
| ------------------------------------ | ----------------------------------------- | -------------------------------------------- |
| Colab GPU = False                    | Training would take hours                 | Changed runtime to T4 GPU                    |
| Roboflow API error                   | Copied placeholder key not real key       | Got actual key from Settings                 |
| [best.pt](http://best.pt) won't open | Tried to open binary file                 | Loaded via YOLO('[best.pt](http://best.pt)') |
| Git push rejected                    | Divergent histories                       | git pull --allow-unrelated-histories         |
| HTTP 400 push error                  | [best.pt](http://best.pt) too large       | Added to .gitignore                          |
| opencv stuck building                | pyproject.toml compilation                | Switched to opencv-python-headless           |
| numpy RuntimeError                   | PyTorch incompatibility                   | pip install numpy<2 --force-reinstall        |
| Translation wrong (akkx)             | Using dot mapping instead of YOLO classes | Read model.names[cls_id] directly            |
| Chat showing raw tuple               | Return value not unpacked                 | reply, history = chat_assistant(...)         |
| IndentationError line 104            | Paste created broken function def         | Removed def detections() line                |
| Black chat input bar                 | CSS not targeting correct element         | Used data-testid selector                    |


---

## Project Structure Built

```
braillevision-2026/
├── app.py                    — Complete Streamlit web app
├── agents.py                 — Three Claude AI agent roles
├── braille_map.py            — A-Z lookup dictionary (tested)
├── best.pt                   — Trained YOLO model (local + Drive)
├── requirements.txt          — Pinned versions
├── setup_instructions.md
├── ai_tools_disclosure.md
├── README.md
├── .streamlit/config.toml    — Brand colors
├── .gitignore
├── model/model_info.md       — Google Drive link for best.pt
├── training/
│   ├── train.ipynb
│   ├── training_logs/results.png
│   └── results/confusion_matrix.png
├── dataset/
│   ├── data.yaml
│   ├── dataset_info.md
│   └── sample_images/        — 5 Braille images
├── inference/inference.py
├── demo/
│   ├── demo_video_link.txt
│   └── screenshots/
└── docs/
    ├── PRD.md
    └── LEARNING_JOURNAL.md
```

---

## What Was Actually Built During Hackathon

### [app.py](http://app.py) — Complete Streamlit Application

- Image upload + camera input
- YOLOv8 inference on uploaded image
- YOLO detection visualization shown alongside input
- Translation displayed reading left-to-right
- gTTS audio playback
- Transcript download as .txt
- Claude context section (with fallback if no credits)
- Ask BrailleBridge chat interface (with fallback if no credits)
- Brand colors `#111111` and `#F97316` applied via CSS

### [agents.py](http://agents.py) — Three Claude Roles

- fallback_verify(): Claude Vision for low-confidence detections
- get_context(): Claude explains what the translation likely is
- chat_assistant(): Conversational Claude with history
- All wrapped in try/except — app never crashes

### inference/[inference.py](http://inference.py) — Standalone Judge Test Script

- Run: python inference/[inference.py](http://inference.py) --source IMAGE --weights [best.pt](http://best.pt)
- Prints all detected letters and confidence scores
- Saves output image to demo/screenshots/

### Complete Documentation

- [README.md](http://README.md) with full setup and run instructions
- setup_[instructions.md](http://instructions.md)
- ai_tools_[disclosure.md](http://disclosure.md)
- dataset/dataset_[info.md](http://info.md)
- model/model_[info.md](http://info.md) with Google Drive link
- training files and logs

---

## Build Log

### May 30 Evening (Pre-hackathon)

- GitHub repo created, scaffold files pushed
- braille_[map.py](http://map.py) written and tested
- YOLO trained on Colab T4 GPU (30 epochs)
- [best.pt](http://best.pt) downloaded and saved locally
- Libraries installed locally

### May 31 (Hackathon Day)

- 4:00PM: Build window opened
- 4:30PM: First official commit pushed
- 5:30PM: Core pipeline working
- 7:00PM: Audio and transcript working
- 7:19PM: App confirmed running with real Braille image
- 7:44PM: Chat CSS fix attempted
- 8:00PM: Phase 1 form prep
- 8:37PM: Translation showing correct letters (ARPQGSTOERNQESX)
- 8:37PM: Chat bug identified and fixed

### June 1 Morning (To Complete)

- Add Anthropic credits
- Test Claude agents fully working
- Take clean screenshots for sample_outputs
- Record demo video
- Edit in capcut
- Complete final submission form

---

## Demo Video Script

0:00-0:08 — Runway AI clip: nurse holding medicine bottle 0:08-0:20 — BrailleBridge title card (Jitter animation) 0:20-0:45 — Problem voiceover 0:45-0:55 — Runway clip: amber dots transforming to letter 0:55-2:30 — Screen recording: full app working 2:30-2:45 — Runway clip: phone scanning bottle 2:45-3:15 — Technical walkthrough 3:15-3:45 — Innovation angle 3:45-4:00 — End card with GitHub link

---

## Final Submission Checklist

```
[ ] Add Anthropic credits (tomorrow morning)
[ ] Test Claude agents fully working
[ ] Screenshot working app with Claude → demo/screenshots/
[ ] git add . && git commit && git push
[ ] Record demo video (4AM-6AM)
[ ] Edit in CapCut
[ ] Upload to YouTube or Google Drive
[ ] Fill final submission form before 5PM June 1
[ ] Confirm GitHub public
[ ] Confirm best.pt Google Drive link works
```

---

## Resources


| Resource                 | URL                                                            |
| ------------------------ | -------------------------------------------------------------- |
| YOLOv8 docs              | [docs.ultralytics.com](http://docs.ultralytics.com)            |
| Dataset                  | universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue |
| Anthropic API            | [docs.anthropic.com](http://docs.anthropic.com)                |
| Streamlit docs           | [docs.streamlit.io](http://docs.streamlit.io)                  |
| Braille alphabet         | pharmabraille.com/braille-alphabet                             |
| Jitter (motion graphics) | [jitter.video](http://jitter.video)                            |
| Runway ML (B-roll)       | [runwayml.com](http://runwayml.com)                            |
| CapCut (video edit)      | mobile app                                                     |


---

## Personal Notes



What was hardest about today: Fixing errors one by one.

What I figured out on my own: Inculcate agents into project 

What surprised me about building this: The commands work the fastest and the built in agents do the work efficiently.

What I would do differently: To add a voice translation and a chatbot in project.

