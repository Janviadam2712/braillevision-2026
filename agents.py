import anthropic


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
        q = user_question.lower()
        if any(w in q for w in ['what', 'mean', 'say', 'read', 'spell']):
            reply = f"The Braille translates to: '{translated_text}'"
        elif any(w in q for w in ['where', 'found', 'object', 'label', 'surface']):
            reply = f"'{translated_text}' in Braille is typically found on books, medicine labels, elevator buttons, or safety signs."
        elif any(w in q for w in ['how', 'work', 'detect', 'yolo', 'model']):
            reply = "BrailleBridge uses YOLOv8 to detect each Braille cell, classifies it by the letter it represents, then reads them left-to-right to form the translation."
        elif any(w in q for w in ['safe', 'medicine', 'drug', 'dose', 'take']):
            reply = "For medical decisions always consult a pharmacist or doctor. BrailleBridge reads the label — it does not provide medical advice."
        elif any(w in q for w in ['hello', 'hi', 'who']):
            reply = "I am BrailleBridge, an AI assistant that translates physical Braille to English. Ask me about any translation."
        else:
            reply = f"BrailleBridge read '{translated_text}' from your Braille image. Ask me what it means, where it might appear, or how the detection works."
        history.append({"role": "assistant", "content": reply})
        return reply, history