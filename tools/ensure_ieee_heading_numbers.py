import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt


TNR = "Times New Roman"

SECTION_TITLES = [
    "INTRODUCTION",
    "RELATED WORK",
    "METHODOLOGY",
    "RESULTS AND ANALYSIS",
    "CONCLUSION",
    "REFERENCES",
]

ROMANS = ["I", "II", "III", "IV", "V"]


def set_run_font(run, size_pt=10, italic=False):
    run.font.name = TNR
    run._element.rPr.rFonts.set(qn("w:ascii"), TNR)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), TNR)
    run.font.size = Pt(size_pt)
    run.font.italic = italic


def rewrite_paragraph(paragraph, text, size_pt=10, italic=False):
    for run in list(paragraph.runs):
        paragraph._element.remove(run._element)
    run = paragraph.add_run(text)
    set_run_font(run, size_pt=size_pt, italic=italic)


def normalize_title(text):
    text = " ".join(text.split())
    text = re.sub(r"^[IVX]+\.\s+", "", text, flags=re.I)
    text = re.sub(r"^[0-9]+\.?\s+", "", text)
    return text.upper()


def normalize_subheading(text):
    text = " ".join(text.split())
    return re.sub(r"^[A-Z]\.\s+", "", text)


def main():
    import sys

    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    doc = Document(src)

    section_seen = 0
    subheading_seen = 0

    for paragraph in doc.paragraphs:
        text = " ".join(paragraph.text.split())
        if not text:
            continue

        normalized = normalize_title(text)
        if normalized in SECTION_TITLES:
            if normalized == "REFERENCES":
                rewrite_paragraph(paragraph, "REFERENCES")
            else:
                if section_seen < len(ROMANS):
                    rewrite_paragraph(paragraph, f"{ROMANS[section_seen]}. {normalized}")
                    section_seen += 1
                    subheading_seen = 0
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(3)
            continue

        if re.match(r"^[A-Z]\.\s+", text) or (
            section_seen > 0 and paragraph.alignment == WD_ALIGN_PARAGRAPH.LEFT and len(text) < 90
        ):
            base = normalize_subheading(text)
            if base and not base.startswith("[") and subheading_seen < 26:
                label = chr(ord("A") + subheading_seen)
                rewrite_paragraph(paragraph, f"{label}. {base}", italic=True)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                subheading_seen += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)


if __name__ == "__main__":
    main()
