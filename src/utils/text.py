"""
Shared text extraction utilities.
"""


def extract_thinking_text(thinking_field):
    """Extract plain text from structured thinking field.

    Handles:
    - list of dicts with 'text' key (structured format from API)
    - plain string
    - other types (converted to str)
    """
    if isinstance(thinking_field, list):
        texts = []
        for item in thinking_field:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif isinstance(item, dict):
                texts.append(item.get("text", ""))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    if isinstance(thinking_field, str):
        return thinking_field
    return str(thinking_field) if thinking_field else ""
