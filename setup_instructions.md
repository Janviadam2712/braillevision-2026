# Setup Instructions

## Requirements
Python 3.9+, webcam or image files

## Installation
git clone https://github.com/Janviadam2712/braillevision-2026.git
cd braillevision-2026
pip install -r requirements.txt

## Download Model
Download best.pt from: PASTE_GOOGLE_DRIVE_LINK_HERE
Place in root: braillevision-2026/best.pt

## Set API Key
export ANTHROPIC_API_KEY="your-key-here"

## Run App
streamlit run app.py
Opens at: http://localhost:8501

## Run Inference Only
python inference/inference.py --source dataset/sample_images/test.jpg --weights best.pt
