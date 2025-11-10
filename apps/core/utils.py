from pathlib import Path
from typing import Union
from pypdf import PdfReader


def extract_text(file_path: Union[str, Path]) -> str:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _extract_from_pdf(file_path)
    elif suffix in {".txt", ".md"}:
        return _extract_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_from_pdf(file_path: Path) -> str:

    try:
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        return _clean_text(text)
    except Exception as e:
        print(f"[WARN] Error reading PDF: {e}")
        return ""


def _extract_from_txt(file_path: Path) -> str:

    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return _clean_text(text)
    except Exception as e:
        print(f"[WARN] Error reading TXT: {e}")
        return ""


def _clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()
