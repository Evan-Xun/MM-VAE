from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt


def set_font(run, name="Times New Roman", east_asia="Songti SC", size=12, bold=False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    run.bold = bold


def add_paragraph(doc, text, size=12, bold=False, first_line_indent=True, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    fmt = p.paragraph_format
    fmt.space_after = Pt(6)
    fmt.line_spacing = 1.5
    if first_line_indent:
        fmt.first_line_indent = Pt(24)
    run = p.add_run(text)
    set_font(run, size=size, bold=bold)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.space_before = Pt(8)
    fmt.space_after = Pt(6)
    run = p.add_run(text)
    set_font(run, size=16 if level == 1 else 14, bold=True)
    return p


def build_doc(output_path: Path):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Songti SC")
    style.font.size = Pt(12)

    section = doc.sections[0]
    section.top_margin = Pt(72)
    section.bottom_margin = Pt(72)
    section.left_margin = Pt(72)
    section.right_margin = Pt(72)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_format = title.paragraph_format
    title_format.space_after = Pt(12)
    run = title.add_run("Methodology章节 3.6 与 3.7")
    set_font(run, size=18, bold=True)

    add_heading(doc, "3.6 MM-VAE模型设计", level=1)
    add_paragraph(
        doc,
        "本文提出的多模态变分自编码器（MM-VAE）以CBC数值特征和红细胞显微图像表示作为输入，通过双编码器结构将两种异构模态映射到共享潜在空间，并在该空间内联合完成重建与分类任务。与简单特征拼接不同，MM-VAE试图在分布层面建模CBC指标与图像形态之间的互补关系，从而学习更具表达能力的患者级表示。",
    )
    add_paragraph(
        doc,
        "设CBC输入表示为xc，图像输入表示为xi。模型分别为两种模态构建独立编码器，将输入映射为潜变量分布参数，即均值mu和对数方差logvar。CBC编码器主要由多层感知机组成，用于处理低维结构化数值特征；图像模态则首先通过卷积神经网络提取患者级图像embedding，再将其输入图像编码器，得到对应的潜变量分布参数。",
    )
    add_paragraph(
        doc,
        "在多模态融合阶段，本文采用Product-of-Experts（PoE）机制对两个模态的后验分布进行联合建模。该机制在高斯分布假设下通过精度加权的方式融合不同模态的信息，使联合后验分布能够更加集中于各模态共享的一致性区域。相较于直接拼接特征向量，PoE能够在潜空间中更显式地刻画跨模态互补性，并天然支持缺失模态场景下的推理。",
    )
    add_paragraph(
        doc,
        "在得到联合潜变量z后，模型包含三个输出分支。第一，CBC重建分支根据z重建原始CBC特征，用于约束潜变量保留结构化血液学信息。第二，图像重建分支根据z重建患者级图像embedding，用于约束潜变量保留红细胞形态学信息。第三，分类分支直接基于z输出贫血二分类结果，用于服务患者级筛查任务。通过这一联合学习框架，模型不仅能够学习判别性表示，还能够借助生成式约束提升共享潜在表示的稳定性和可解释性。",
    )
    add_paragraph(
        doc,
        "此外，为适配实际临床场景中可能出现的数据不完整问题，MM-VAE在推理阶段支持三种输入模式：完整模态输入、缺失图像模态输入以及缺失CBC模态输入。当某一模态缺失时，模型仍可利用剩余模态编码得到潜变量分布，并完成分类预测。该设计使MM-VAE不仅适用于标准多模态融合任务，也能够用于评估模态缺失条件下的性能退化与鲁棒性。",
    )

    add_heading(doc, "3.7 损失函数与训练策略", level=1)
    add_paragraph(
        doc,
        "MM-VAE的训练目标由四部分组成，分别是CBC重建损失、图像重建损失、KL散度损失以及分类损失。为避免Word中公式和特殊符号出现乱码，本文采用纯文本形式表示总体损失：",
        first_line_indent=False,
    )
    add_paragraph(
        doc,
        "总损失 = CBC重建损失 + 图像重建权重 × 图像重建损失 + KL权重 × KL散度损失 + 分类权重 × 分类损失",
        size=12,
        bold=True,
        first_line_indent=False,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    add_paragraph(
        doc,
        "其中，CBC重建损失表示对CBC模态重建误差的约束，图像重建损失表示对图像模态重建误差的约束，KL散度损失用于约束潜变量后验分布接近先验分布，分类损失用于直接优化贫血二分类任务。图像重建权重、KL权重和分类权重分别用于平衡这三部分目标在总损失中的贡献。",
    )
    add_paragraph(
        doc,
        "在具体实现中，CBC重建损失与图像重建损失均采用均方误差进行计算，以度量模型重建结果与原始输入之间的偏差。KL散度项用于将联合潜变量分布约束到标准正态先验附近，从而避免潜空间过于发散，并促进不同样本在共享表示空间中的结构化分布。分类损失则采用带logits的二元交叉熵损失，直接优化模型在贫血筛查任务上的判别性能。",
    )
    add_paragraph(
        doc,
        "为了避免训练初期KL约束过强导致潜变量学习不稳定，本文采用KL warm-up策略，即在训练前若干轮逐步提升KL项的权重，使模型先学习基本的重建与分类能力，再逐步加强潜空间正则化。该策略有助于缓解后验塌缩问题，并提升多模态表示学习的稳定性。根据当前实验设置，KL项权重在前20个epoch内线性增长，随后保持稳定。",
    )
    add_paragraph(
        doc,
        "在优化方法上，本文采用Adam优化器对模型参数进行更新。所有神经网络模型均在训练集上进行优化，并根据验证集性能选择最优checkpoint。验证阶段以ROC-AUC为优先模型选择指标；若某些设置下无法稳定计算ROC-AUC，则退而使用F1-score作为辅助选择标准。为降低过拟合风险，训练过程中采用早停策略，当验证集指标在连续若干轮内不再提升时终止训练。当前项目的主要设置为最大训练轮数100轮，早停耐心值15轮。",
    )
    add_paragraph(
        doc,
        "在实现层面，MM-VAE同时优化生成式目标与判别式目标，因此其训练过程可以视为一种带监督约束的多模态表示学习方案。与单纯追求分类精度的拼接式融合方法相比，该方案能够通过重建约束和分布约束提升潜变量的语义一致性，从而为后续的消融分析、缺失模态测试以及潜空间解释提供更稳定的基础。",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)


if __name__ == "__main__":
    build_doc(Path("artifacts/Methodology_3.6_3.7.docx"))
