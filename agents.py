# agents.py — BrailleBridge AI Agents
# Three Claude-powered roles:
# Role 1: Fallback Verifier  → when YOLO confidence < 0.6
# Role 2: Context Layer      → explains what was translated
# Role 3: Chat Assistant     → answers follow-up questions
# Full implementation happens during hackathon

import anthropic
import base64
import cv2

def fallback_verify(image_crop_bgr, yolo_confidence, threshold=0.6):
    """Role 1: Claude Vision verifies uncertain Braille dots."""
    pass  # Implement during hackathon

def get_context(translated_text: str) -> str:
    """Role 2: Claude explains what the translated text likely is."""
    pass  # Implement during hackathon

def chat_assistant(user_question: str, translated_text: str, history: list):
    """Role 3: Claude answers follow-up questions about the translation."""
    pass  # Implement during hackathon