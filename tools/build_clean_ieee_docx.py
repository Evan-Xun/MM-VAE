from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


SRC = Path("artifacts/Anemia_Screening_MMVAE_reviewer_revised_layout.docx")
OUT = Path("artifacts/Anemia_Screening_MMVAE_reviewer_revised_IEEE_clean.docx")
FIGURES = {
    "fig1": Path("artifacts/overleaf_ieee/figures/image1.png"),
    "fig2": Path("artifacts/overleaf_ieee/figures/image2.png"),
    "fig3": Path("artifacts/fig_table4_1_main_comparison.png"),
    "fig4": Path("artifacts/fig_table4_2_hgb_ablation.png"),
    "fig5": Path("artifacts/fig_table4_3_missing_modality.png"),
}


TNR = "Times New Roman"


def set_run(run, size, bold=False, italic=False):
    run.font.name = TNR
    run._element.rPr.rFonts.set(qn("w:ascii"), TNR)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), TNR)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), TNR)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic


def add_text(doc, text, size=10, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=0):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(text)
    set_run(run, size, bold=bold, italic=italic)
    return p


def set_margins(section):
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(0.625)
    section.right_margin = Inches(0.625)
    section.header_distance = Inches(0.5)
    section.footer_distance = Inches(0.5)


def set_columns(section, num=2, space_twips=300):
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))


def add_footer(section):
    section.different_first_page_header_footer = True
    p = section.first_page_footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.clear()
    run = p.add_run("XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE")
    set_run(run, 8)


def add_author_cell(cell, name, affiliation):
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(name)
    set_run(r, 10)
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(affiliation)
    set_run(r, 9, italic=True)


def add_author_block(doc):
    table = doc.add_table(rows=2, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row in table.rows:
        for cell in row.cells:
            cell.width = Inches(2.35)

    common = "School of Computing and Data Science\nXiamen University Malaysia\nSepang, 43900, Malaysia"
    add_author_cell(table.cell(0, 0), "Fengxun Liu", common + "\nSWE2309525@xmu.edu.my")
    add_author_cell(table.cell(0, 1), "Yuxuan Huang", common + "\nSWE2309515@xmu.edu.my")
    add_author_cell(table.cell(0, 2), "Zeyuan Li", common + "\nSWE2309523@xmu.edu.my")
    merged = table.cell(1, 0).merge(table.cell(1, 2))
    add_author_cell(
        merged,
        "Xiaolin Li*",
        "Department of Clinical Laboratory\nThe Second Affiliated Hospital of Nanchang University\n"
        "Nanchang, Jiangxi, China\nndefy21158@ncu.edu.cn",
    )

    for row in table.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                element = OxmlElement(f"w:{edge}")
                element.set(qn("w:val"), "nil")
                borders.append(element)
            tc_pr.append(borders)
    add_text(doc, "*Corresponding author.", size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)


def add_section_heading(doc, text, roman):
    label = f"{roman}. {text.upper()}" if roman else text.upper()
    p = add_text(doc, label, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=3)
    p.paragraph_format.space_before = Pt(6)


def add_subheading(doc, text, letter):
    p = add_text(doc, f"{letter}. {text}", size=10, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=1)
    p.paragraph_format.space_before = Pt(3)


def add_figure(doc, path, caption, width=3.35):
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = add_text(doc, caption, size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    cap.paragraph_format.keep_with_next = False


def add_experiment_table(doc):
    add_text(doc, "TABLE I", size=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_text(doc, "Experimental design summary.", size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    rows = [
        ["ID", "Model", "CBC", "Image", "HGB", "Purpose"],
        ["G0", "Rule-based", "HGB only", "N/A", "Yes", "Clinical baseline"],
        ["G1", "CBC-only", "core_with_hgb", "N/A", "Yes", "Unimodal (with HGB)"],
        ["G2", "CBC-only", "core_without_hgb", "N/A", "No", "Unimodal (without HGB)"],
        ["G3", "Image-only", "N/A", "Embedding", "No", "Unimodal image"],
        ["G4", "Concat Fusion", "core_without_hgb", "Embedding", "No", "Fusion baseline (no HGB)"],
        ["G5", "MM-VAE", "core_without_hgb", "Embedding", "No", "Proposed (no HGB)"],
        ["G6", "Concat Fusion", "core_with_hgb", "Embedding", "Yes", "Fusion baseline (with HGB)"],
        ["G7", "MM-VAE", "core_with_hgb", "Embedding", "Yes", "Proposed (with HGB)"],
    ]
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(value)
            set_run(r, 6.5, bold=(i == 0))


def normalized_paragraphs():
    doc = Document(SRC)
    vals = [" ".join(p.text.split()) for p in doc.paragraphs]
    vals = [v for v in vals if v]
    return vals


def main():
    source = normalized_paragraphs()
    doc = Document()
    set_margins(doc.sections[0])
    add_footer(doc.sections[0])

    title = source[0]
    add_text(doc, title, size=22, align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
    add_author_block(doc)

    abstract = source[10] + " " + source[11]
    keywords = source[12]
    add_text(doc, abstract.replace("Abstract—", "Abstract—"), size=9, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=4)
    add_text(doc, keywords.replace("Index Terms—", "Index Terms—"), size=9, italic=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6)

    body = doc.add_section(WD_SECTION.CONTINUOUS)
    set_margins(body)
    set_columns(body, 2)
    add_footer(body)

    roman = {"Introduction": "I", "Related Work": "II", "Methodology": "III", "Results And Analysis": "IV", "Conclusion": "V"}
    subheads = {
        "Computer-aided detection of anemia",
        "Erythrocyte microscopy image analysis",
        "VAE and multimodal representation learning",
        "Research framework",
        "Data preparation and preprocessing",
        "Model design",
        "Training strategy, experimental setup, and evaluation met-rics",
        "Main comparison experiment",
        "HGB ablation experiment",
        "Missing-modality robustness experiment",
    }
    sub_counter = 0
    current_section = None
    skip = {"TABLE I", "Experimental design summary."}

    for text in source[13:]:
        if text in skip or text.startswith("Fig. "):
            continue
        if text in roman:
            current_section = text
            sub_counter = 0
            add_section_heading(doc, text, roman[text])
            continue
        if text == "References":
            add_section_heading(doc, text, "")
            continue
        if current_section and text in subheads:
            if text == "Data preparation and preprocessing":
                add_figure(doc, FIGURES["fig1"], "Fig. 1. Overall framework of the proposed patient-level anemia screening system (Photo credit: Original).")
            if text == "Training strategy, experimental setup, and evaluation met-rics":
                add_figure(doc, FIGURES["fig2"], "Fig. 2. Architecture of the proposed MM-VAE model (Photo credit: Original).")
            add_subheading(doc, text, chr(ord("A") + sub_counter))
            sub_counter += 1
            continue
        if text == "(1)":
            add_text(doc, text, size=10, align=WD_ALIGN_PARAGRAPH.RIGHT)
            continue
        if text.startswith("The experimental design is summarized in Table I"):
            add_text(doc, text, size=10)
            add_experiment_table(doc)
            continue
        if text.startswith("The following 5 methods were compared"):
            add_text(doc, text, size=10)
            add_figure(doc, FIGURES["fig3"], "Fig. 3. Results under the core_with_hgb setting (Photo credit: Original).")
            continue
        if text.startswith("Removing HGB from the CBC input"):
            add_text(doc, text, size=10)
            add_figure(doc, FIGURES["fig4"], "Fig. 4. Performance under with_hgb and without_hgb configurations (Photo credit: Original).")
            continue
        if text.startswith("A further test is whether"):
            add_text(doc, text, size=10)
            add_figure(doc, FIGURES["fig5"], "Fig. 5. Missing-modality inference results of MM-VAE under core_with_hgb (Photo credit: Original).")
            continue
        if current_section == "References":
            add_text(doc, text, size=8, align=WD_ALIGN_PARAGRAPH.LEFT)
        else:
            add_text(doc, text, size=10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
