import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt


SRC = Path("/Users/fengxun/Downloads/main.tex")
FIG_DIR = Path("/Users/fengxun/Downloads/figures")
OUT = Path("artifacts/Anemia_Screening_MMVAE_IEEE_revised.docx")


FIG_LABELS = {
    "fig:framework": 1,
    "fig:mmvvae": 2,
    "fig:main-results": 3,
    "fig:hgb-ablation": 4,
    "fig:missing-modality": 5,
    "fig:missing-rate": 6,
}

TABLE_LABELS = {
    "tab:experiment-design": 1,
}

EXP_TABLE = [
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


def set_margins(section):
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(1.57)
    section.right_margin = Cm(1.57)


def set_columns(section, count=2, space_twips=360):
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    cols = cols[0] if cols else OxmlElement("w:cols")
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(space_twips))
    if not sect_pr.xpath("./w:cols"):
        sect_pr.append(cols)


def style_doc(doc):
    for section in doc.sections:
        set_margins(section)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.0

    for style_name, size, bold in [
        ("Heading 1", 10, True),
        ("Heading 2", 10, True),
    ]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = bold


def add_paragraph(doc, text, align=None, bold=False, italic=False, size=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return p


def clean_latex(text):
    text = text.replace("\\c{S}", "S").replace("\\c{s}", "s")
    text = text.replace("\\c{c}", "c").replace("\\c{C}", "C")
    text = text.replace("{\\i}", "i").replace("\\i", "i")
    text = text.replace("\\%", "%").replace("\\_", "_")
    text = text.replace("--", "-")
    text = text.replace("~", " ")
    text = text.replace("\\times", "x").replace("\\pm", "±")
    text = text.replace("\\alpha", "alpha").replace("\\beta", "beta")
    text = text.replace("\\lambda", "lambda")
    text = re.sub(r"\$([^$]*)\$", lambda m: clean_latex(m.group(1)), text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\cite\{([^}]*)\}", cite_repl, text)
    text = re.sub(r"\\ref\{([^}]*)\}", ref_repl, text)
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{([^}]*)\})?", lambda m: m.group(1) or "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cite_repl(match):
    keys = [k.strip() for k in match.group(1).split(",")]
    nums = [str(BIB_ORDER[k]) for k in keys if k in BIB_ORDER]
    return "[" + ", ".join(nums) + "]"


def ref_repl(match):
    key = match.group(1)
    if key in FIG_LABELS:
        return str(FIG_LABELS[key])
    if key in TABLE_LABELS:
        return str(TABLE_LABELS[key])
    if key == "eq:loss":
        return "1"
    return key


def extract_block(text, begin, end):
    return text.split(begin, 1)[1].split(end, 1)[0].strip()


def add_heading(doc, text, level):
    p = doc.add_paragraph()
    p.style = f"Heading {level}"
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(8 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text.upper() if level == 1 else text)
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(10)
    run.bold = True


def add_figure(doc, filename, caption, full_width=True):
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    set_margins(section)
    set_columns(section, 1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    width = Inches(6.8 if full_width else 3.3)
    p.add_run().add_picture(str(FIG_DIR / filename), width=width)
    cap = add_paragraph(doc, caption, align=WD_ALIGN_PARAGRAPH.CENTER, size=8)
    cap.paragraph_format.space_after = Pt(8)
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    set_margins(section)
    set_columns(section, 2)


def add_experiment_table(doc):
    doc.add_page_break()
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    set_margins(section)
    set_columns(section, 1)
    add_paragraph(doc, "Table 1. Experimental design summary.", align=WD_ALIGN_PARAGRAPH.CENTER, size=8)
    table = doc.add_table(rows=len(EXP_TABLE), cols=len(EXP_TABLE[0]))
    table.style = "Table Grid"
    widths = [0.55, 1.2, 1.5, 1.0, 0.55, 2.0]
    for r, row in enumerate(EXP_TABLE):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.width = Inches(widths[c])
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER if c in [0, 3, 4] else WD_ALIGN_PARAGRAPH.LEFT
            run = para.add_run(val)
            run.font.name = "Times New Roman"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
            run.font.size = Pt(7.5)
            run.bold = r == 0
    doc.add_paragraph()
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    set_margins(section)
    set_columns(section, 2)


def add_equation(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("L = L_cbc + alpha L_img + beta L_kl + lambda L_cls     (1)")
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run.italic = True


def parse_body(text):
    body = text.split("\\section{Introduction}", 1)[1].split("\\begin{thebibliography}", 1)[0]
    body = "\\section{Introduction}" + body
    body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", figure_token, body, flags=re.S)
    body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "\n[[TABLE:experiment]]\n", body, flags=re.S)
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "\n[[EQUATION:loss]]\n", body, flags=re.S)
    return body


def figure_token(match):
    block = match.group(0)
    img = re.search(r"\\includegraphics(?:\[[^]]*\])?\{figures/(image\d+\.png)\}", block).group(1)
    caption = clean_latex(re.search(r"\\caption\{([^}]*)\}", block, flags=re.S).group(1))
    label = re.search(r"\\label\{([^}]*)\}", block)
    num = FIG_LABELS.get(label.group(1), len(FIG_LABELS)) if label else len(FIG_LABELS)
    return f"\n[[FIGURE:{img}:Figure {num}. {caption}]]\n"


def add_body(doc, body):
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("\\section{"):
            add_heading(doc, clean_latex(line[9:-1]), 1)
        elif line.startswith("\\subsection{"):
            add_heading(doc, clean_latex(line[12:-1]), 2)
        elif line.startswith("[[FIGURE:"):
            _, img, caption = line.strip("[]").split(":", 2)
            add_figure(doc, img, caption, full_width=(img != "image5.png"))
        elif line == "[[TABLE:experiment]]":
            add_experiment_table(doc)
        elif line == "[[EQUATION:loss]]":
            add_equation(doc)
        elif line.startswith("\\FloatBarrier") or line in ["\\centering", "\\hline"]:
            continue
        else:
            text = clean_latex(line)
            if text:
                add_paragraph(doc, text)


def add_references(doc, text):
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    set_margins(section)
    set_columns(section, 2)
    add_heading(doc, "References", 1)
    refs = re.findall(r"\\bibitem\{([^}]*)\}\s*(.*?)(?=\n\\bibitem|\n\\end\{thebibliography\})", text, flags=re.S)
    for i, (_, ref) in enumerate(refs, 1):
        add_paragraph(doc, f"[{i}] {clean_latex(ref)}", size=8)


def main():
    text = SRC.read_text(encoding="utf-8")
    global BIB_ORDER
    bibs = re.findall(r"\\bibitem\{([^}]*)\}", text)
    BIB_ORDER = {key: i + 1 for i, key in enumerate(bibs)}

    doc = Document()
    style_doc(doc)
    set_columns(doc.sections[0], 1)

    title = clean_latex(re.search(r"\\title\{([^}]*)\}", text).group(1))
    add_paragraph(doc, title, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=18)

    authors = [
        ("Fengxun Liu", "School of Computing and Data Science, Xiamen University Malaysia, Sepang, 43900, Malaysia", "SWE2309525@xmu.edu.my"),
        ("Yuxuan Huang", "School of Computing and Data Science, Xiamen University Malaysia, Sepang, 43900, Malaysia", "SWE2309515@xmu.edu.my"),
        ("Zeyuan Li", "School of Computing and Data Science, Xiamen University Malaysia, Sepang, 43900, Malaysia", "SWE2309523@xmu.edu.my"),
        ("Xiaolin Li*", "Department of Clinical Laboratory, The Second Affiliated Hospital of Nanchang University, Nanchang, Jiangxi 330006, China", "ndefy21158@ncu.edu.cn"),
    ]
    for name, aff, email in authors:
        add_paragraph(doc, name, align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
        add_paragraph(doc, aff, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9)
        add_paragraph(doc, email, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)
    add_paragraph(doc, "*Corresponding author.", size=8)

    abstract = clean_latex(extract_block(text, "\\begin{abstract}", "\\end{abstract}"))
    add_paragraph(doc, "Abstract-" + abstract, bold=False, size=9)
    keywords = clean_latex(extract_block(text, "\\begin{IEEEkeywords}", "\\end{IEEEkeywords}"))
    add_paragraph(doc, "Keywords-" + keywords, size=9)

    section = doc.add_section(WD_SECTION.CONTINUOUS)
    set_margins(section)
    set_columns(section, 2)
    add_body(doc, parse_body(text))
    add_references(doc, text)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
