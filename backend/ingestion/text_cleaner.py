"""
Text cleaning and normalization utility for extracted PDF content.
"""
import re


def clean_text(raw_text: str) -> str:
    """
    Cleans and standardizes raw text extracted from PDF documents.
    - Strips non-printable ASCII control characters.
    - Recombines hyphenated word wraps across lines.
    - Normalizes excessive whitespace and redundant empty lines.
    - Preserves meaningful paragraph structure and technical symbols.
    """
    if not raw_text:
        return ""

    # Replace null bytes and non-printable control characters except standard whitespace
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", raw_text)

    # Recombine hyphenated line breaks (e.g., 'trans- \nformer' -> 'transformer')
    cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", cleaned)

    # Convert Windows CRLF and CR to standard LF
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Replace multiple spaces/tabs on a single line with a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Collapse 3 or more consecutive newlines into 2 (paragraph break)
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

    # Strip leading and trailing whitespace from each line
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    return cleaned.strip()


def detect_scanned_page(text: str, min_chars: int = 50) -> bool:
    """
    Determines if an extracted page has insufficient characters to be valid text,
    suggesting a scanned image or non-OCR document.
    """
    visible_chars = len(re.sub(r"\s+", "", text or ""))
    return visible_chars < min_chars


def detect_tables_and_equations(text: str) -> tuple[bool, bool]:
    """
    Heuristically checks if text contains tabular data (repeated columns/tabs/bars)
    or mathematical equations.
    """
    has_tables = bool(
        re.search(r"(\|[^\n]+\|)|(\b\w+\s{3,}\w+\s{3,}\w+)", text)
    )
    has_equations = bool(
        re.search(r"(\b[a-zA-Z]\s*=\s*[\d\w\+\-\*/\^]+)|\b(sum|int|lim|prod)\b", text)
    )
    return has_tables, has_equations
