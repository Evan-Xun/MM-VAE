from pathlib import Path
import json

from docx import Document


def section_info(section):
    cols = None
    sect_pr = section._sectPr
    cols_el = sect_pr.xpath("./w:cols")
    if cols_el:
        cols = {
            "num": cols_el[0].get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}num"),
            "space": cols_el[0].get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}space"),
        }
    return {
        "start_type": str(section.start_type),
        "page_width": section.page_width,
        "page_height": section.page_height,
        "top_margin": section.top_margin,
        "bottom_margin": section.bottom_margin,
        "left_margin": section.left_margin,
        "right_margin": section.right_margin,
        "header_distance": section.header_distance,
        "footer_distance": section.footer_distance,
        "columns": cols,
    }


def para_info(p, idx):
    text = " ".join(p.text.split())
    return {
        "index": idx,
        "style": p.style.name if p.style else None,
        "alignment": str(p.alignment),
        "text": text[:220],
        "run_fonts": list(
            {
                (
                    run.font.name,
                    run.font.size.pt if run.font.size else None,
                    run.bold,
                    run.italic,
                )
                for run in p.runs
                if run.text.strip()
            }
        ),
    }


def main():
    import sys

    path = Path(sys.argv[1])
    doc = Document(path)
    payload = {
        "sections": [section_info(s) for s in doc.sections],
        "paragraph_count": len(doc.paragraphs),
        "first_paragraphs": [para_info(p, i) for i, p in enumerate(doc.paragraphs[:60])],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
