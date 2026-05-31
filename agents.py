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
    """Role 2: Claude explains what the translated text likely is."""
    text = (translated_text or "").strip()
    if not text:
        return "Translate Braille first to see contextual explanation."

    client = _get_client()
    if client is None:
        return (
            "AI context is disabled. Set ANTHROPIC_API_KEY to get explanations "
            "of what this translation might mean in everyday life."
        )

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You help sighted caregivers understand Braille translations. "
                        f"In 2–3 short sentences, explain what this English text likely "
                        f"refers to (medicine label, sign, name, etc.): \"{text}\""
                    ),
                }
            ],
        )
        return response.content[0].text
    except Exception as exc:
        return f"Could not load context: {exc}"


def chat_assistant(user_question: str, translated_text: str, history: list):
    """Role 3: Claude answers follow-up questions about the translation."""
    question = (user_question or "").strip()
    if not question:
        return "Please enter a question."

    client = _get_client()
    if client is None:
        return "Set ANTHROPIC_API_KEY to use the chat assistant."

    messages = []
    for turn in history or []:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    context = (translated_text or "").strip() or "(no translation yet)"
    messages.append(
        {
            "role": "user",
            "content": (
                f"Current Braille translation: \"{context}\"\n\n"
                f"Follow-up question: {question}"
            ),
        }
    )

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=400,
            system=(
                "You are BrailleBridge, a helpful assistant for Braille translation. "
                "Answer briefly and clearly about the translation and Braille literacy."
            ),
            messages=messages,
        )
        return response.content[0].text
    except Exception as exc:
        return f"Sorry, I could not answer: {exc}"
