from pathlib import Path

from docx import Document
from docx.shared import Pt


SRC = Path("/Users/fengxun/Downloads/Anemia_Screening_MMVAE.docx")
OUT = Path("artifacts/Anemia_Screening_MMVAE_reviewer_revised_content.docx")


def set_times_new_roman(paragraph, size_pt=10):
    for run in paragraph.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(size_pt)


def replace_paragraph_starting(doc, prefix, text):
    for paragraph in doc.paragraphs:
        if " ".join(paragraph.text.split()).startswith(prefix):
            paragraph.clear()
            paragraph.add_run(text)
            set_times_new_roman(paragraph)
            return True
    return False


def replace_inline_everywhere(doc, replacements):
    containers = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                containers.extend(cell.paragraphs)

    for paragraph in containers:
        for run in paragraph.runs:
            text = run.text
            for old, new in replacements.items():
                text = text.replace(old, new)
            run.text = text


def main():
    doc = Document(SRC)

    replace_inline_everywhere(
        doc,
        {
            "cbcmask": "cbc_mask",
            "corewithhgb": "core_with_hgb",
            "corewithouthgb": "core_without_hgb",
            "withhgb": "with_hgb",
            "withouthgb": "without_hgb",
            "missingimage": "missing_image",
            "missingcbc": "missing_cbc",
        },
    )

    replace_paragraph_starting(
        doc,
        "For the training strategy, the objective function of MM-VAE consists of",
        (
            "For the training strategy, the objective function of MM-VAE consists of "
            "CBC reconstruction loss, image embedding reconstruction loss, KL divergence loss, "
            "and classification loss. Its overall form is given as:"
        ),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
