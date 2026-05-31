# BrailleBridge

> Point. Read. Understand. — Real-time Braille to English translation using AI.

## Problem

Physical Braille appears on medicine bottles, safety signs, and elevator buttons. Sighted caregivers and nurses who encounter these objects daily cannot read them. BrailleBridge solves this instantly with any camera.

## Demo

[Video link — to be added June 1]

## How It Works

1. Upload image or take photo of physical Braille

2. OpenCV preprocesses the image

3. YOLOv8 detects and classifies each Braille cell 

4. Translation displayed as text 

5. gTTS converts to spoken audio 

6. Claude AI provides context and answers follow-up questions

## Architecture

Camera/Image → OpenCV → YOLOv8 ([best.pt](http://best.pt)) → Letter mapping → Text output + Audio + Transcript + Claude AI context + Chat

## Tech Stack

- YOLOv8n (fine-tuned, 1324 images, 26 classes) 

- OpenCV-headless 

- Streamlit 

- gTTS 

- Anthropic Claude API

## Setup

git clone [https://github.com/Janviadam2712/braillevision-2026](https://github.com/Janviadam2712/braillevision-2026) 

cd braillevision-2026 

pip install -r requirements.txt 

Download [best.pt](http://best.pt): [YOUR GOOGLE DRIVE LINK] 

Place [best.pt](http://best.pt) in root directory 

export ANTHROPIC_API_KEY="your-key" 

streamlit run [app.py](http://app.py) 

Open: [http://localhost:8501](http://localhost:8501)

## Dataset

Roboflow Braille Detection V2 

URL: [https://universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue](https://universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue) 

License: CC BY 4.0 | Images: 1324 | Classes: 26 (A-Z)

## Model

YOLOv8n fine-tuned for 30 epochs on T4 GPU 

[best.pt](http://best.pt) download: [YOUR GOOGLE DRIVE LINK]

## AI Tools Disclosure

See ai_tools_[disclosure.md](http://disclosure.md)

## Judging Criteria

1. Accuracy: YOLOv8 with 0.90+ confidence on Braille cells 

2. Real-time: Inference under 200ms per image 

3. Technical: Full pipeline from camera to audio output 

4. Robustness: Tested on dataset images and real physical Braille 

5. Accessibility: Audio output + transcript + AI chat 

6. Innovation: 3 Claude AI agents for verification, context, chat 

7. Demo: See video link above

## Author

Janvi | BrailleVision Hackathon 2026 | Solo submission