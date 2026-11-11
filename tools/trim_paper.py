"""
Trim the paper from ~4100 words to ~3600 words by removing verbose/redundant passages.
Cuts (targeting ~500 words):
  1. Intro "Against this background" paragraph → condensed (~80w saved)
  2. Related Work 2.1 third paragraph → condensed (~50w saved)
  3. Related Work 2.3 "In summary" paragraph → deleted (~100w saved)
  4. Results 4.1 Image-only caveat → condensed (~50w saved)
  5. Conclusion future work → condensed (~90w saved)
  6. Conclusion limitations → condensed (~50w saved)
  7. Methodology 3.3 experiment description → condensed (~60w saved)
"""

from docx import Document
from docx.oxml.ns import qn
import copy, re

SRC = "/Users/fengxun/studyspace/科研项目/科研写作/约课2作业.docx"
DST = "/Users/fengxun/studyspace/科研项目/科研写作/约课2作业_trimmed.docx"

doc = Document(SRC)

def para_text(p):
    return "".join(r.text for r in p.runs)

def set_para_text(p, new_text):
    """Replace all runs in paragraph with a single run carrying the first run's format."""
    if not p.runs:
        return
    fmt_run = p.runs[0]
    # Remove all runs
    for r in p.runs:
        r._element.getparent().remove(r._element)
    # Add single run with new text
    import copy
    new_r = copy.deepcopy(fmt_run._element)
    # Clear text nodes and set new text
    for t in new_r.findall(qn('w:t')):
        new_r.remove(t)
    from docx.oxml import OxmlElement
    t_elem = OxmlElement('w:t')
    t_elem.text = new_text
    if new_text.startswith(' ') or new_text.endswith(' '):
        t_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    new_r.append(t_elem)
    p._element.append(new_r)

def delete_para(p):
    p._element.getparent().remove(p._element)

paras = doc.paragraphs

# Build index of paragraphs with their text
indexed = [(i, para_text(p).strip()) for i, p in enumerate(paras)]

def find_para(start_text, start=0):
    """Find paragraph index by matching start of text."""
    for i, (idx, txt) in enumerate(indexed):
        if i < start:
            continue
        if txt.startswith(start_text[:60]):
            return i
    return -1

# ── Cut 1: Condense "Against this background" paragraph ────────────────────
i = find_para("Against this background")
if i >= 0:
    set_para_text(paras[i],
        "Against this background, this study investigates whether an MM-VAE "
        "combining VAE-based latent-space regularization with a supervised "
        "classification objective can outperform unimodal and late-fusion baselines, "
        "and whether it learns genuine cross-modal complementarity beyond reliance on "
        "a single dominant feature such as HGB.")
    print(f"Cut 1 applied at para {i}")

# ── Cut 2: Condense Related Work 2.1 third paragraph ───────────────────────
i = find_para("However, the existing anemia detection methods are still insufficient")
if i >= 0:
    set_para_text(paras[i],
        "However, existing methods remain limited. CBC-based approaches rely heavily "
        "on strong indicators such as HGB, while image-based methods are constrained "
        "by image quality and limited generalization. Crucially, the complementary "
        "relationship between CBC reports and blood smear microscopy images has not "
        "been systematically modeled at the patient level.")
    print(f"Cut 2 applied at para {i}")

# ── Cut 3: Delete "In summary" paragraph in Related Work 2.3 ───────────────
i = find_para("In summary, the existing multimodal fusion method")
if i >= 0:
    delete_para(paras[i])
    print(f"Cut 3 applied at para {i}")

# Re-index after deletion
paras = doc.paragraphs
indexed = [(i, para_text(p).strip()) for i, p in enumerate(paras)]

# ── Cut 4: Condense Image-only caveat in Results 4.1 ───────────────────────
i = find_para("The Image-only result needs a caveat")
if i >= 0:
    set_para_text(paras[i],
        "The Image-only model collapsed into an all-positive classifier "
        "(recall 1.0000, specificity 0.0000), consistent with the limited "
        "standalone discriminative capacity of the frozen mean-pooled embedding "
        "noted in Section 3.2. This limitation resurfaces in Section 4.3.")
    print(f"Cut 4 applied at para {i}")

# ── Cut 5: Condense Methodology 3.3 experiment description ─────────────────
i = find_para("As summarized in Table 1, the experimental design")
if i >= 0:
    set_para_text(paras[i],
        "As summarized in Table 1, the experimental design consists of three parts: "
        "a main comparison experiment across all models, an HGB ablation experiment "
        "testing two feature configurations (with_hgb and without_hgb) to verify "
        "genuine cross-modal complementarity, and a missing-modality robustness "
        "experiment evaluating three inference modes (complete, missing_image, "
        "missing_cbc) within a single unified MM-VAE without retraining.")
    print(f"Cut 5 applied at para {i}")

# ── Cut 6: Condense Conclusion limitations paragraph ───────────────────────
i = find_para("This study remains limitations")
if i >= 0:
    set_para_text(paras[i],
        "This study has several limitations. The frozen mean-pooled image "
        "representation limits standalone image discriminative power. "
        "Experiments were conducted on a single dataset, and label approximation "
        "error may be introduced due to the use of a uniform HGB threshold without "
        "sex-stratified severity analysis.")
    print(f"Cut 6 applied at para {i}")

# ── Cut 7: Condense future work paragraph ──────────────────────────────────
i = find_para("Future work may proceed in three directions")
if i >= 0:
    set_para_text(paras[i],
        "Future work includes exploring fine-tuned image encoders or patch-level "
        "attention mechanisms to strengthen image-modality discriminative capacity, "
        "external validation on larger multi-center datasets, and latent-variable "
        "correlation analyses to improve interpretability and clinical relevance.")
    print(f"Cut 7 applied at para {i}")

doc.save(DST)
print(f"\nSaved to: {DST}")

# Count words
full_text = " ".join(para_text(p) for p in doc.paragraphs)
words = len(full_text.split())
print(f"Estimated word count: {words}")
