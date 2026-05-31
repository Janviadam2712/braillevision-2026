# BrailleBridge

> Point. Read. Understand. — Real-time Braille to English translation using AI.

## Problem

Physical Braille appears on medicine bottles, safety signs, and everyday objects.

Sighted caregivers, nurses, and teachers who encounter these objects daily cannot

read them. BrailleBridge solves this with a camera, YOLO, and Claude AI.

## Demo

[Video link — to be added June 1]

## How It Works

Camera input → OpenCV preprocessing → YOLOv8 dot detection →

Braille mapping → English text + audio + transcript + Claude AI context

## Tech Stack

- YOLOv8n (fine-tuned on Braille Detection V2)

- OpenCV (image preprocessing)

- Streamlit (web UI)

- gTTS (text-to-speech)

- Anthropic Claude API (3 AI agent roles)

## Setup

pip install -r requirements.txt

Download [best.pt](http://best.pt) from: [GOOGLE DRIVE LINK]

export ANTHROPIC_API_KEY="your-key"

streamlit run [app.py](http://app.py)

## Dataset

Roboflow Braille Detection V2 — CC BY 4.0

[https://universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue](https://universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue)

1,324 images | 26 classes (A-Z)

## AI Tools Disclosure

See ai_tools_[disclosure.md](http://disclosure.md)

## Author

Janvi | BrailleVision Hackathon 2026