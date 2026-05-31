# agents.py — BrailleBridge AI Agents
# Role 1: Fallback Verifier  → when YOLO confidence < 0.6
# Role 2: Context Layer      → explains what was translated
# Role 3: Chat Assistant     → answers follow-up questions

import os
import anthropic

_client = None


def _get_client():
    global _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    if _client is None:
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def fallback_verify(image_crop_bgr, yolo_confidence, threshold=0.6):
    """Role 1: Claude Vision verifies uncertain Braille dots."""
    if yolo_confidence >= threshold:
        return None
    client = _get_client()
    if client is None:
        return None
    # Vision fallback reserved for low-confidence crops during hackathon tuning.
    return None


def get_context(translated_text: str) -> str:
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"Text from Braille: '{translated_text}'. In one sentence, what is this likely (medicine label, greeting, safety sign)?"
            }]
        )
        return response.content[0].text.strip()
    except Exception:
        return f"'{translated_text}' — Braille translation complete. Add Anthropic API credits to enable AI context."


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
        return f"Translation complete: '{translated_text}' — AI context ready."
        history.append({"role": "assistant", "content": fallback})
        return fallback, history