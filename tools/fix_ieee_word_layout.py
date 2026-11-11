import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


DEFAULT_SRC = Path("/Users/fengxun/Downloads/Anemia_Screening_MMVAE.docx")
DEFAULT_OUT = Path("artifacts/Anemia_Screening_MMVAE_IEEE_layout_fixed.docx")


def set_text(paragraph, text, *, size=None, italic=False, bold=False):
    paragraph.clear()
    for idx, line in enumerate(text.split("\n")):
        if idx:
            paragraph.add_run().add_break()
        run = paragraph.add_run(line)
        font = run.font
        font.name = "Times New Roman"
        font.size = Pt(size) if size else None
        font.italic = italic
        font.bold = bold


def replace_in_runs(doc, replacements):
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            text = run.text
            for src, dst in replacements.items():
                text = text.replace(src, dst)
            run.text = text
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        text = run.text
                        for src, dst in replacements.items():
                            text = text.replace(src, dst)
                        run.text = text


def replace_in_paragraph_text(doc, replacements):
    containers = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                containers.extend(cell.paragraphs)

    for paragraph in containers:
        text = paragraph.text
        new_text = text
        for src, dst in replacements.items():
            new_text = new_text.replace(src, dst)
        if new_text != text:
            alignment = paragraph.alignment
            style = paragraph.style
            paragraph.clear()
            run = paragraph.add_run(new_text)
            run.font.name = "Times New Roman"
            if style.name == "Body Text":
                run.font.size = Pt(10)
            elif paragraph.text.strip().startswith("["):
                run.font.size = Pt(7.5)
            paragraph.style = style
            paragraph.alignment = alignment


def remove_paragraph(paragraph):
    element = paragraph._element
    element.getparent().remove(element)
    paragraph._p = paragraph._element = None


def merge_split_intro_and_move_corresponding_note(doc):
    paragraphs = doc.paragraphs

    # Put the corresponding-author explanation in the author block rather than
    # in the middle of the first Introduction paragraph.
    author_name_idx = next(
        i for i, paragraph in enumerate(paragraphs) if paragraph.text.strip() == "Xiaolin Li*"
    )
    note_idx = author_name_idx + 2
    if note_idx < len(paragraphs):
        paragraphs[note_idx].alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraphs[note_idx].paragraph_format.left_indent = Inches(0)
        paragraphs[note_idx].paragraph_format.right_indent = Inches(0)
        set_text(paragraphs[note_idx], "*Corresponding author.", size=8)
        paragraphs[note_idx].paragraph_format.space_after = Pt(8)

    paragraphs = doc.paragraphs
    for i, paragraph in enumerate(list(paragraphs)):
        if i < 15 or paragraph.text.strip() != "*Corresponding author.":
            continue
        if i > 0 and i + 1 < len(paragraphs):
            previous = paragraphs[i - 1]
            following = paragraphs[i + 1]
            merged = f"{previous.text.rstrip()} {following.text.lstrip()}"
            previous.clear()
            run = previous.add_run(merged)
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            previous.style = following.style
            previous.alignment = following.alignment
            remove_paragraph(following)
        remove_paragraph(paragraph)
        break


def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)

    for edge, attrs in kwargs.items():
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        for key, value in attrs.items():
            element.set(qn(f"w:{key}"), str(value))


def clean_table_borders(table):
    border = {"val": "single", "sz": "8", "space": "0", "color": "000000"}
    none = {"val": "nil"}

    for row_idx, row in enumerate(table.rows):
        for cell in row.cells:
            set_cell_border(cell, top=none, bottom=none, left=none, right=none)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(8)
        if row_idx == 0:
            for cell in row.cells:
                set_cell_border(cell, top=border, bottom=border)
        if row_idx == len(table.rows) - 1:
            for cell in row.cells:
                set_cell_border(cell, bottom=border)


def style_references(doc):
    in_refs = False
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().lower() == "references":
            in_refs = True
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue
        if not in_refs:
            continue
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(7.5)


def fix_author_block(doc):
    paragraphs = doc.paragraphs

    # In the original Word conversion, the corresponding author name was inside
    # the three-column author section, which made it drop under the first author.
    for paragraph in list(paragraphs):
        if paragraph.text.strip() == "Xiaolin Li*":
            remove_paragraph(paragraph)
            break

    paragraphs = doc.paragraphs
    dept_idx = next(
        i
        for i, paragraph in enumerate(paragraphs)
        if paragraph.text.strip() == "Department of Clinical Laboratory"
    )

    paragraphs[dept_idx].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraphs[dept_idx].paragraph_format.left_indent = Inches(0)
    paragraphs[dept_idx].paragraph_format.right_indent = Inches(0)
    set_text(paragraphs[dept_idx], "Xiaolin Li*", size=12)
    paragraphs[dept_idx].paragraph_format.space_before = Pt(6)
    paragraphs[dept_idx].paragraph_format.space_after = Pt(0)

    affil = (
        "Department of Clinical Laboratory\n"
        "The Second Affiliated Hospital of Nanchang University\n"
        "Nanchang, Jiangxi, China\n"
        "ndefy21158@ncu.edu.cn"
    )
    paragraphs[dept_idx + 1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraphs[dept_idx + 1].paragraph_format.left_indent = Inches(0)
    paragraphs[dept_idx + 1].paragraph_format.right_indent = Inches(0)
    set_text(paragraphs[dept_idx + 1], affil, size=10, italic=True)
    paragraphs[dept_idx + 1].paragraph_format.space_after = Pt(8)


def main():
    import sys

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)

    doc = Document(out)
    fix_author_block(doc)
    merge_split_intro_and_move_corresponding_note(doc)
    replacements = {
        "corewithhgb": "core_with_hgb",
        "corewithouthgb": "core_without_hgb",
        "core with hgb": "core_with_hgb",
        "core without hgb": "core_without_hgb",
        "withhgb": "with_hgb",
        "withouthgb": "without_hgb",
        "with hgb": "with_hgb",
        "without hgb": "without_hgb",
        "cbcmask": "cbc_mask",
        "missingimage": "missing_image",
        "missingcbc": "missing_cbc",
        "missing image": "missing_image",
        "missing cbc": "missing_cbc",
        "Kılıc¸": "Kilic",
        "Kılıc": "Kilic",
        "Kilic¸": "Kilic",
        "S¸ .": "S.",
        "S¸.": "S.",
    }
    replace_in_runs(doc, replacements)
    replace_in_paragraph_text(doc, replacements)

    if doc.tables:
        clean_table_borders(doc.tables[0])
    style_references(doc)

    doc.save(out)
    print(out)


if __name__ == "__main__":
    main()
