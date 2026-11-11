from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


TNR = "Times New Roman"


def set_run_font(run, size_pt, bold=False, italic=False):
    run.font.name = TNR
    run._element.rPr.rFonts.set(qn("w:ascii"), TNR)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), TNR)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), TNR)
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic


def set_paragraph_common(p, size_pt=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p.alignment = align
    fmt = p.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.line_spacing = 1.0
    for run in p.runs:
        if run.text.strip():
            set_run_font(
                run,
                size_pt=size_pt,
                bold=bool(run.bold),
                italic=bool(run.italic),
            )


def clear_paragraph_runs(paragraph):
    for run in list(paragraph.runs):
        paragraph._element.remove(run._element)


def format_title(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(12)
    fmt.line_spacing = 1.0
    for run in paragraph.runs:
        if run.text.strip():
            set_run_font(run, 24)


def format_corresponding(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(8)
    fmt.line_spacing = 1.0
    for run in paragraph.runs:
        if run.text.strip():
            set_run_font(run, 8)


def format_abstract_like(paragraph, label, body, label_bold=True):
    clear_paragraph_runs(paragraph)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(4)
    fmt.line_spacing = 1.0
    run = paragraph.add_run(label)
    set_run_font(run, 9, bold=label_bold, italic=True)
    run = paragraph.add_run(body)
    set_run_font(run, 9)


def format_section_heading(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(6)
    fmt.space_after = Pt(3)
    fmt.line_spacing = 1.0
    text = paragraph.text.upper()
    clear_paragraph_runs(paragraph)
    run = paragraph.add_run(text)
    set_run_font(run, 10)


def apply_template_footer(section):
    section.different_first_page_header_footer = True
    footer = section.first_page_footer
    for p in footer.paragraphs:
        clear_paragraph_runs(p)

    if not footer.paragraphs:
        p = footer.add_paragraph()
    else:
        p = footer.paragraphs[0]

    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt = p.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.line_spacing = 1.0
    run = p.add_run("XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE")
    set_run_font(run, 8)

    # Keep subsequent page footers quiet, matching the template's submission style.
    for p in section.footer.paragraphs:
        clear_paragraph_runs(p)


def format_subheading(paragraph):
    text = paragraph.text
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(3)
    fmt.space_after = Pt(1)
    fmt.line_spacing = 1.0
    clear_paragraph_runs(paragraph)
    run = paragraph.add_run(text)
    set_run_font(run, 10, italic=True)


def set_hanging_indent(paragraph, left=0.22, hanging=0.22):
    fmt = paragraph.paragraph_format
    fmt.left_indent = Inches(left)
    fmt.first_line_indent = Inches(-hanging)


def add_columns_to_sectpr(sect_pr, num=2, space_twips=300):
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))


def insert_continuous_two_col_break(paragraph):
    p_pr = paragraph._element.get_or_add_pPr()
    sect_pr = OxmlElement("w:sectPr")
    type_el = OxmlElement("w:type")
    type_el.set(qn("w:val"), "continuous")
    sect_pr.append(type_el)
    pg_sz = OxmlElement("w:pgSz")
    pg_sz.set(qn("w:w"), "12240")
    pg_sz.set(qn("w:h"), "15840")
    sect_pr.append(pg_sz)
    pg_mar = OxmlElement("w:pgMar")
    pg_mar.set(qn("w:top"), "1080")
    pg_mar.set(qn("w:right"), "900")
    pg_mar.set(qn("w:bottom"), "1440")
    pg_mar.set(qn("w:left"), "900")
    pg_mar.set(qn("w:header"), "720")
    pg_mar.set(qn("w:footer"), "720")
    pg_mar.set(qn("w:gutter"), "0")
    sect_pr.append(pg_mar)
    add_columns_to_sectpr(sect_pr, num=2, space_twips=300)
    p_pr.append(sect_pr)


def format_table(table):
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is not None:
        tbl_w.set(qn("w:type"), "dxa")
        tbl_w.set(qn("w:w"), "9360")
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                fmt = p.paragraph_format
                fmt.space_before = Pt(0)
                fmt.space_after = Pt(0)
                fmt.line_spacing = 1.0
                for run in p.runs:
                    if run.text.strip():
                        bold = bool(run.bold)
                        set_run_font(run, 8.5, bold=bold)


def main():
    import sys

    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    doc = Document(src)

    for section in doc.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(0.625)
        section.right_margin = Inches(0.625)
        section.header_distance = Inches(0.5)
        section.footer_distance = Inches(0.5)
        apply_template_footer(section)

    body_sect = doc.sections[-1]._sectPr
    add_columns_to_sectpr(body_sect, num=2, space_twips=300)

    for table in doc.tables:
        format_table(table)

    for i, p in enumerate(doc.paragraphs):
        text = " ".join(p.text.split())
        if i == 0:
            format_title(p)
        elif i == 1:
            format_corresponding(p)
        elif text.startswith("Abstract—"):
            format_abstract_like(p, "Abstract—", text[len("Abstract—"):])
        elif text.startswith("Keywords—"):
            format_abstract_like(p, "Keywords—", text[len("Keywords—"):], label_bold=False)
        elif text in {
            "I. Introduction",
            "II. Related Work",
            "III. Methodology",
            "IV. Results and Analysis",
            "V. Conclusion",
            "References",
        }:
            if text == "References":
                format_section_heading(p)
            else:
                format_section_heading(p)
        elif len(text) > 3 and text[1:3] == ". " and text[0].isalpha():
            format_subheading(p)
        elif text.startswith("[") and "]" in text[:5]:
            set_paragraph_common(p, size_pt=8)
            set_hanging_indent(p, left=0.20, hanging=0.20)
        elif text.startswith("Figure ") or text.startswith("Table "):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt = p.paragraph_format
            fmt.space_before = Pt(2)
            fmt.space_after = Pt(4)
            fmt.line_spacing = 1.0
            for run in p.runs:
                if run.text.strip():
                    set_run_font(run, 8)
        elif text == "(1)":
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            for run in p.runs:
                if run.text.strip():
                    set_run_font(run, 10)
        elif text:
            set_paragraph_common(p, size_pt=10)

    ref_idx = next((i for i, p in enumerate(doc.paragraphs) if " ".join(p.text.split()) == "References"), None)
    if ref_idx is not None and ref_idx > 0:
        insert_continuous_two_col_break(doc.paragraphs[ref_idx - 1])

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)


if __name__ == "__main__":
    main()
