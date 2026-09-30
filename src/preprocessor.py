import re
import unicodedata

DOMAIN_SUFFIXES = ('.com', '.org', '.net', '.in', '.co', '.info', '.io', '.fr', '.biz', '.us')
NUM_REGEX = re.compile(r"\b\d[\w]*\b")

def normalize_text(text: str) -> str:
    """
    Standardize raw entity text:
    1. NFKC Unicode normalization.
    2. Lowercasing.
    3. Retain alphanumeric tokens, ASCII characters, and international Unicode scripts (Indic, French, etc.).
    4. Collapse duplicate whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    x = unicodedata.normalize("NFKC", str(text)).lower()
    out = []
    for ch in x:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        elif ord(ch) < 128:
            out.append(" ")
        elif unicodedata.category(ch).startswith(("L", "M", "N", "Z")):
            out.append(ch)
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()

def get_numeric_tokens_and_key(norm_addr: str) -> tuple[list[str], str]:
    """
    Extracts numerical identifiers (building numbers, postal codes, suite numbers)
    from normalized address strings and generates a sorted pipe-delimited key.
    Example: '85 wayne ave ny 10001' -> (['10001', '85'], '10001|85')
    """
    if not norm_addr:
        return [], ""
    tokens = sorted(set(NUM_REGEX.findall(norm_addr)))
    return tokens, "|".join(tokens)

def clean_domain(text: str) -> str:
    """
    Strips top-level domain extensions (.com, .in, etc.) for URL-based entity names.
    """
    t = text.lower().strip()
    for suff in DOMAIN_SUFFIXES:
        if t.endswith(suff):
            t = t[:-len(suff)]
            break
    return t

def get_char_3grams(text: str) -> set[str]:
    """
    Generates character trigram set for fuzzy n-gram overlap.
    """
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}

def preprocess_record(name: str, address: str) -> tuple[str, str, str]:
    """
    High-level entity preprocessor returning (norm_name, norm_address, numeric_key).
    """
    norm_n = normalize_text(name)
    norm_a = normalize_text(address)
    _, num_key = get_numeric_tokens_and_key(norm_a)
    return norm_n, norm_a, num_key
