import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str):
    # Regex-based word tokenizer
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
