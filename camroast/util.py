# camroast/util.py
"""Small helpers shared by several modules."""


def short_err(e: BaseException) -> str:
    """A one-line, human-sized version of an API error (OpenAI, ElevenLabs, or anything else)."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            msg = detail.get("message") or detail.get("status")
            if msg:
                return str(msg)
        if isinstance(detail, str):
            return detail
    msg = getattr(e, "message", None)
    if isinstance(msg, str) and msg:
        return msg.splitlines()[0][:160]
    text = str(e).strip() or e.__class__.__name__
    return text.splitlines()[0][:160]
