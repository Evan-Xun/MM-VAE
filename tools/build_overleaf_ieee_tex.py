import re
import shutil
import zipfile
from pathlib import Path

from docx import Document


SRC_DOCX = Path("artifacts/PACSMV-404014-G6-conference-template-letter-numbered.docx")
OUT_DIR = Path("artifacts/overleaf_ieee")
FIG_DIR = OUT_DIR / "figures"
MAIN_TEX = OUT_DIR / "main.tex"

FIGURES = {
    1: ("image1.png", "fig:framework"),
    2: ("image2.png", "fig:mmvvae"),
    3: ("image3.png", "fig:main-results"),
    4: ("image4.png", "fig:hgb-ablation"),
    5: ("image5.png", "fig:missing-modality"),
}

DOI_MAP = {
    "1": "10.3389/fpubh.2022.964385",
    "2": "10.3390/electronics9030427",
    "3": "10.1093/database/baae120",
    "4": "10.1080/00051144.2024.2352317",
    "5": "10.4258/hir.2025.31.1.57",
    "6": "10.1007/s10278-025-01479-6",
    "7": "10.1515/bmt-2021-0127",
    "8": "10.48550/arXiv.1406.5298",
    "9": "10.48550/arXiv.1802.05335",
    "10": "10.48550/arXiv.2403.13658",
}


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "–": "--",
        "—": "---",
        "−": "-",
        "μ": r"$\mu$",
        "Ş": r"\c{S}",
        "ı": r"{\i}",
        "ç": r"\c{c}",
        "±": r"$\pm$",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def citeify(text):
    def repl(match):
        nums = re.findall(r"\d+", match.group(1))
        if not nums:
            return match.group(0)
        return r"\cite{" + ",".join(f"ref{n}" for n in nums) + "}"

    return re.sub(r"\[([0-9,\s]+)\]", repl, text)


def crossref(text):
    mappings = {
        "Figure 1": r"Fig.~\ref{fig:framework}",
        "Figure 2": r"Fig.~\ref{fig:mmvvae}",
        "Figure 3": r"Fig.~\ref{fig:main-results}",
        "Figure 4": r"Fig.~\ref{fig:hgb-ablation}",
        "Figure 5": r"Fig.~\ref{fig:missing-modality}",
        "Table 1": r"Table~\ref{tab:experiment-design}",
    }
    for src, dst in mappings.items():
        text = text.replace(src, dst)
    return text


def body_text(text):
    return crossref(citeify(latex_escape(text)))


def clean_heading(text):
    text = " ".join(text.split())
    text = re.sub(r"^[IVX]+\.\s+", "", text)
    return text.title()


def clean_subheading(text):
    return re.sub(r"^[A-Z]\.\s+", "", " ".join(text.split()))


def figure_block(number, caption):
    filename, label = FIGURES[number]
    if number == 5:
        return (
            "\\begin{figure}[!t]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\columnwidth]{{figures/{filename}}}\n"
            f"\\caption{{{body_text(caption)}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{figure}\n"
        )

    width = r"\textwidth" if number in {1, 2, 3, 4} else r"\columnwidth"
    return (
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.92{width}]{{figures/{filename}}}\n"
        f"\\caption{{{body_text(caption)}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure*}\n"
    )


def table_block(table):
    rows = [[cell.text.replace("\n", " ").strip() for cell in row.cells] for row in table.rows]
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Experimental design summary.}",
        "\\label{tab:experiment-design}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{p{0.06\\textwidth}p{0.15\\textwidth}p{0.18\\textwidth}p{0.12\\textwidth}p{0.08\\textwidth}p{0.28\\textwidth}}",
        "\\hline",
        " & ".join(rf"\textbf{{{latex_escape(v)}}}" for v in rows[0]) + r" \\",
        "\\hline",
    ]
    for row in rows[1:]:
        lines.append(" & ".join(latex_escape(v) for v in row) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def parse_references(ref_paragraphs):
    bibitems = []
    for text in ref_paragraphs:
        match = re.match(r"\[(\d+)\]\s*(.*)", text)
        if not match:
            continue
        num, body = match.groups()
        doi = DOI_MAP.get(num)
        if doi:
            body = f"{body}, doi: {doi}."
        bibitems.append(f"\\bibitem{{ref{num}}} {latex_escape(body)}")
    return "\n".join(bibitems)


def extract_media():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SRC_DOCX) as zf:
        for number, (filename, _) in FIGURES.items():
            data = zf.read(f"word/media/{filename}")
            (FIG_DIR / filename).write_bytes(data)


def build_tex():
    doc = Document(SRC_DOCX)
    paragraphs = [" ".join(p.text.split()) for p in doc.paragraphs]
    table = doc.tables[1]

    title = paragraphs[0]
    abstract = paragraphs[3].removeprefix("Abstract--").removeprefix("Abstract—").strip()
    keywords = paragraphs[4].removeprefix("Keywords--").removeprefix("Keywords—").strip()

    out = []
    references = []
    in_references = False

    for i, text in enumerate(paragraphs[5:], start=5):
        if not text:
            continue
        if text == "REFERENCES":
            in_references = True
            continue
        if in_references:
            references.append(text)
            continue
        if re.match(r"^[IVX]+\.\s+", text):
            if text.startswith("V."):
                out.append("\\FloatBarrier\n")
            out.append(f"\\section{{{latex_escape(clean_heading(text))}}}\n")
            continue
        if re.match(r"^[A-Z]\.\s+", text):
            out.append(f"\\subsection{{{latex_escape(clean_subheading(text))}}}\n")
            continue
        fig_match = re.match(r"Figure\s+(\d+)\.\s+(.*)", text)
        if fig_match:
            out.append(figure_block(int(fig_match.group(1)), fig_match.group(2)))
            continue
        if text.startswith("Table 1."):
            out.append(table_block(table))
            continue
        if text == "(1)":
            out.append(
                "\\begin{equation}\n"
                "\\mathcal{L}=\\mathcal{L}_{cbc}+\\alpha\\mathcal{L}_{img}+\\beta\\mathcal{L}_{kl}+\\lambda\\mathcal{L}_{cls}\n"
                "\\label{eq:loss}\n"
                "\\end{equation}\n"
            )
            continue
        if text.startswith("Here, , , and"):
            text = (
                "Here, the image reconstruction loss, KL divergence loss, and classification loss "
                "respectively constrain image embedding reconstruction, regularization of the shared latent space, "
                "and anemia binary classification performance. Earlier versions of the framework assigned equal "
                "weight to all CBC dimensions during reconstruction. The current implementation instead applies "
                "a mask-based reconstruction loss for the CBC branch, computed using cbc_mask. This means "
                "reconstruction error is calculated only over the CBC indicators that were actually observed. "
                "This prevents the model from being overly driven by the imputed values themselves during training. "
                "All learning models are trained with the Adam optimizer, with early stopping based on validation "
                "set performance. To stabilize the learning of the latent variable, MM-VAE applies a KL warm-up "
                "strategy in the early stages of training, gradually increasing the weight of the KL term."
            )
        out.append(body_text(text) + "\n")

    tex = rf"""\documentclass[conference]{{IEEEtran}}
\IEEEoverridecommandlockouts
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{algorithmic}}
\usepackage{{graphicx}}
\usepackage{{textcomp}}
\usepackage{{xcolor}}
\usepackage{{placeins}}
\def\BibTeX{{{{\rm B\kern-.05em{{\sc i\kern-.025em b}}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{{E}}\kern-.125emX}}}}

\begin{{document}}

\title{{{latex_escape(title)}}}

\author{{\IEEEauthorblockN{{Fengxun Liu}}
\IEEEauthorblockA{{\textit{{School of Computing and Data Science}} \\
\textit{{Xiamen University Malaysia}}\\
Sepang, 43900, Malaysia \\
SWE2309525@xmu.edu.my}}
\and
\IEEEauthorblockN{{Xiaolin Li\textsuperscript{{*}}}}
\IEEEauthorblockA{{\textit{{Department of Clinical Laboratory}} \\
\textit{{the Second Affiliated Hospital of Nanchang University}}\\
Nanchang, Jiangxi, China \\
ndefy21158@ncu.edu.cn}}
}}

\maketitle
\begingroup
\renewcommand\thefootnote{{}}
\footnotetext{{\textsuperscript{{*}}Corresponding author.}}
\endgroup

\begin{{abstract}}
{body_text(abstract)}
\end{{abstract}}

\begin{{IEEEkeywords}}
{body_text(keywords)}
\end{{IEEEkeywords}}

{chr(10).join(out)}

\FloatBarrier
\begin{{thebibliography}}{{00}}
{parse_references(references)}
\end{{thebibliography}}

\end{{document}}
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_TEX.write_text(tex, encoding="utf-8")


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    extract_media()
    build_tex()


if __name__ == "__main__":
    main()
