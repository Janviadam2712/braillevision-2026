import anthropic
import base64
import cv2

def fallback_verify(image_crop_bgr, yolo_confidence, threshold=0.6):
    if yolo_confidence >= threshold:
        return None
    try:
        client = anthropic.Anthropic()
        _, buffer = cv2.imencode('.jpg', image_crop_bgr)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64
                    }},
                    {"type": "text",
                     "text": "This is a Braille cell. Which letter? Reply with only the letter."}
                ]
            }]
        )
        return response.content[0].text.strip().lower()
    except Exception:
        return None


def get_context(translated_text: str) -> str:
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"Text from Braille: '{translated_text}'. In one sentence what is this likely — medicine label, greeting, safety sign, book?"
            }]
        )
        return response.content[0].text.strip()
    except Exception:
        # Smart rule-based fallback — no credits needed
        text = translated_text.lower()
        if len(translated_text) <= 2:
            return f"'{translated_text}' — short Braille code or abbreviation detected."
        elif any(c in text for c in ['mg', 'ml', 'rx']):
            return f"'{translated_text}' — this appears to be a pharmaceutical or medicine label."
        elif any(c in text for c in ['exit', 'stop', 'warn', 'caut']):
            return f"'{translated_text}' — this appears to be a safety or warning sign."
        elif text in ['hello', 'hi', 'bye', 'yes', 'no', 'ok']:
            return f"'{translated_text}' — common greeting or response phrase."
        else:
            return f"'{translated_text}' — Braille translation complete. Common on books, signs, or product labels."


def chat_assistant(user_question: str, translated_text: str, history: list):
    try:
        client = anthropic.Anthropic()
        system = f"""You are an accessibility assistant.
Current Braille translation: '{translated_text}'.
Answer questions briefly and helpfully."""
        history.append({"role": "user", "content": user_question})
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system,
            messages=history
        )
        reply = response.content[0].text.strip()
        history.append({"role": "assistant", "content": reply})
        return reply, history
    except Exception:
        # Smart keyword-based fallback
        q = user_question.lower()
        if any(w in q for w in ['what', 'mean', 'say', 'read']):
            reply = f"The Braille text reads: '{translated_text}'"
        elif any(w in q for w in ['object', 'where', 'found', 'label']):
            reply = f"'{translated_text}' is commonly found on medicine bottles, safety signs, books, or product packaging."
        elif any(w in q for w in ['safe', 'medicine', 'drug', 'dose']):
            reply = "For medical information always consult a healthcare professional."
        elif any(w in q for w in ['how', 'work', 'detect', 'yolo']):
            reply = "BrailleBridge uses YOLOv8 to detect Braille characters and reads them left to right."
        else:
            reply = f"Translation: '{translated_text}'. Ask me what it means or where it might appear."
        history.append({"role": "assistant", "content": reply})
        return reply, history