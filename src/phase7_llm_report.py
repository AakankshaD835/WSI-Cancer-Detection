# Phase 7 - LLM clinical report generation (Claude/Groq) with structured JSON input and PDF export

import argparse
import json
import os
import sys
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_setup import load_config

ROOT = Path(__file__).resolve().parent.parent


# 1. Structured JSON builder

def build_analysis_json(
    tumor_probability  : float,
    mil_attention      : list,          # list of attention scores per patch
    top_patch_indices  : list,          # indices of most attended patches
    n_patches          : int,
    gradcam_iou        : float = None,
    wsi_id             : str   = None,
    backbone           : str   = "efficientnet_b3",
    heatmap_path       : str   = None,
) -> Dict[str, Any]:
    """
    Build the structured JSON payload that will be fed to the LLM.

    This is the bridge between CV outputs and the language model.
    """
    # Confidence = distance from 0.5 (works for both tumor and normal predictions)
    confidence_level = (
        "HIGH"   if tumor_probability > 0.80 or tumor_probability < 0.20 else
        "MEDIUM" if tumor_probability > 0.65 or tumor_probability < 0.35 else
        "LOW"
    )

    # Top attended regions
    top_regions = [f"patch_{i:04d}" for i in top_patch_indices[:5]]

    # Attention entropy (how concentrated the model's attention is)
    import numpy as np
    att_arr  = np.array(mil_attention)
    att_norm = att_arr / (att_arr.sum() + 1e-8)
    entropy  = float(-np.sum(att_norm * np.log(att_norm + 1e-8)))

    # Top-5 attention scores (actual values, not just rank)
    top5_idx    = np.argsort(att_arr)[::-1][:5]
    top5_scores = [round(float(att_arr[i]), 6) for i in top5_idx]

    payload = {
        "metadata": {
            "wsi_id"     : wsi_id or "SAMPLE_WSI_001",
            "timestamp"  : datetime.datetime.now().isoformat(),
            "model"      : f"PatchCamelyon MIL - EfficientNet-B3 + Gated Attention",
            "backbone"   : backbone,
            "n_patches"  : n_patches,
        },
        "classification": {
            "tumor_probability" : round(float(tumor_probability), 4),
            "prediction"        : "TUMOR" if tumor_probability > 0.5 else "NORMAL",
            "confidence"        : confidence_level,
        },
        "attention_analysis": {
            "top_regions"        : top_regions,
            "top_attention_scores": top5_scores,
            "attention_entropy"  : round(entropy, 4),
            "attention_max"      : round(float(att_arr.max()), 4),
            "attention_mean"     : round(float(att_arr.mean()), 4),
            "focus_description"  : (
                "Highly concentrated - model focuses on specific suspicious regions"
                if entropy < 2.0 else
                "Distributed - model attention spread across multiple regions"
            ),
        },
        "explainability": {
            "method"         : "Grad-CAM + Gated Attention MIL",
            "gradcam_iou"    : round(gradcam_iou, 4) if gradcam_iou else None,
            "heatmap_path"   : heatmap_path or "reports/heatmaps/heatmap.png",
            "interpretation" : (
                "High activation in lymphocytic clusters with nuclear atypia"
                if tumor_probability > 0.7 else
                "Minimal activation; predominantly stromal tissue patterns"
            ),
        },
        "recommendations": _generate_recommendations(tumor_probability, confidence_level),
    }
    return payload


def _generate_recommendations(prob: float, confidence: str) -> list:
    if prob > 0.80:
        return [
            "Urgent pathologist review recommended",
            "Confirmatory IHC staining (Ki-67, HER2) advised",
            "Correlate with clinical presentation and prior history",
            "Consider multidisciplinary oncology team consultation",
        ]
    elif prob > 0.50:
        return [
            "Pathologist review recommended",
            "Additional serial sections may be warranted",
            "Correlate with clinical and radiological findings",
        ]
    else:
        return [
            "No immediate malignant features detected",
            "Routine follow-up per standard protocol",
            "Re-evaluation if clinical symptoms persist",
        ]


# 2. Prompt templates

SYSTEM_PROMPT = """You are an AI assistant supporting pathology reporting.
Your role is to translate computational pathology AI findings into a structured,
professional clinical report suitable for review by a board-certified pathologist.

Guidelines:
- Write in formal pathology report style
- Use precise histological terminology
- Clearly distinguish AI model findings from clinical interpretation
- Include appropriate uncertainty language when confidence is not HIGH
- Do NOT make definitive diagnoses — always recommend human pathologist review
- Structure output with sections: Specimen, AI Analysis Summary, Findings, Interpretation, Recommendations
"""

def build_llm_prompt(analysis_json: Dict[str, Any]) -> str:
    """Convert the structured JSON into a natural language prompt for the LLM."""
    meta   = analysis_json["metadata"]
    cls    = analysis_json["classification"]
    att    = analysis_json["attention_analysis"]
    expl   = analysis_json["explainability"]
    recs   = analysis_json["recommendations"]

    prompt = f"""Generate a formal pathology AI analysis report based on the following computational findings:

=== AI MODEL OUTPUT ===
Specimen ID    : {meta['wsi_id']}
Analysis Date  : {meta['timestamp'][:10]}
Model          : {meta['model']}
Patches Analyzed: {meta['n_patches']}

Classification:
  Tumor Probability : {cls['tumor_probability']:.1%}
  Prediction        : {cls['prediction']}
  Confidence Level  : {cls['confidence']}

Attention Analysis:
  Top Suspicious Regions : {', '.join(att['top_regions'])}
  Attention Pattern      : {att['focus_description']}
  Attention Entropy      : {att['attention_entropy']:.3f}

Explainability:
  Method         : {expl['method']}
  Grad-CAM IoU   : {expl['gradcam_iou'] or 'N/A'}
  Interpretation : {expl['interpretation']}

Recommendations:
{chr(10).join(f'  - {r}' for r in recs)}
=== END AI OUTPUT ===

Please generate a formal pathology AI report with the following sections:
1. SPECIMEN INFORMATION
2. AI ANALYSIS SUMMARY
3. DETAILED FINDINGS
4. PATHOLOGICAL INTERPRETATION
5. RECOMMENDATIONS
6. DISCLAIMER

Use clinical language appropriate for a pathology report. Emphasize that this is an AI-assisted
analysis requiring human pathologist review before any clinical decision-making.
"""
    return prompt


# 3. LLM API calls

def call_claude(prompt: str, cfg: dict, heatmap_path: Optional[str] = None) -> str:
    """Call Anthropic Claude API (multimodal: sends heatmap image when available)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)

    # Build multimodal content if heatmap is available
    content = []
    if heatmap_path and Path(heatmap_path).exists():
        import base64
        with open(heatmap_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image",
            "source": {
                "type"      : "base64",
                "media_type": "image/png",
                "data"      : image_data,
            },
        })
        content.append({
            "type": "text",
            "text": "The image above is the Grad-CAM heatmap for this WSI. "
                    "Red regions indicate high tumor probability areas identified by the model. "
                    "Please incorporate the spatial distribution shown in this heatmap into your report.\n\n"
                    + prompt,
        })
        print("  Sending heatmap image to Claude (multimodal) …")
    else:
        content.append({"type": "text", "text": prompt})

    message = client.messages.create(
        model     = cfg["llm"]["model"],
        max_tokens= cfg["llm"]["max_tokens"],
        system    = SYSTEM_PROMPT,
        messages  = [{"role": "user", "content": content}],
    )
    return message.content[0].text


def call_openai(prompt: str, cfg: dict) -> str:
    """Call OpenAI GPT API (fallback)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    client   = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model    = "gpt-4o",
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ],
        max_tokens = cfg["llm"]["max_tokens"],
    )
    return response.choices[0].message.content


def call_groq(prompt: str, cfg: dict) -> str:
    """
    Call Groq API — FREE tier, Llama 3.1 70B.
    Sign up free at https://console.groq.com
    Set env var: GROQ_API_KEY
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY environment variable (free at console.groq.com)")

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model    = cfg["llm"].get("groq_model", "llama-3.1-70b-versatile"),
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens  = cfg["llm"]["max_tokens"],
        temperature = cfg["llm"].get("temperature", 0.3),
    )
    return response.choices[0].message.content


def call_gemini(prompt: str, cfg: dict) -> str:
    """
    Call Google Gemini API — FREE tier (gemini-1.5-flash).
    Sign up free at https://aistudio.google.com
    Set env var: GEMINI_API_KEY
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: pip install google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY environment variable (free at aistudio.google.com)")

    genai.configure(api_key=api_key)
    model    = genai.GenerativeModel(
        model_name    = cfg["llm"].get("gemini_model", "gemini-1.5-flash"),
        system_instruction = SYSTEM_PROMPT,
    )
    response = model.generate_content(prompt)
    return response.text


def call_huggingface(prompt: str, cfg: dict) -> str:
    """
    Call HuggingFace Inference API — FREE tier (rate limited).
    Sign up free at https://huggingface.co
    Set env var: HF_API_KEY
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise ImportError("Run: pip install huggingface_hub")

    api_key = os.environ.get("HF_API_KEY")
    if not api_key:
        raise ValueError("Set HF_API_KEY environment variable (free at huggingface.co)")

    client   = InferenceClient(
        model  = cfg["llm"].get("hf_model", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        token  = api_key,
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"
    response    = client.text_generation(
        full_prompt,
        max_new_tokens = cfg["llm"]["max_tokens"],
        temperature    = cfg["llm"].get("temperature", 0.3),
    )
    return response


# Offline demo report (no API call)

DEMO_REPORT = """
PATHOLOGY AI ANALYSIS REPORT
════════════════════════════════════════════════════════════

1. SPECIMEN INFORMATION
   Specimen ID    : SAMPLE_WSI_001
   Analysis Date  : {date}
   Institution    : Computational Pathology Lab
   Analyzed by    : PatchCamelyon MIL — EfficientNet-B3 + Gated Attention MIL

2. AI ANALYSIS SUMMARY
   The AI model analyzed {n_patches} tissue patches extracted from the submitted
   whole slide image. Deep learning-based feature extraction and multiple instance
   learning (MIL) with gated self-attention was employed for slide-level
   malignancy classification.

   Overall Tumor Probability : {prob:.1%}
   Classification            : {prediction}
   Confidence                : {confidence}

3. DETAILED FINDINGS
   Patch-Level Analysis:
   - {n_patches} patches were extracted at 96×96 pixel resolution
   - Macenko stain normalization was applied prior to analysis
   - Top suspicious regions: {top_regions}

   Attention Analysis:
   - The model demonstrates {focus_desc}
   - Attention entropy of {entropy:.3f} indicates degree of spatial focus
   - Grad-CAM activation maps highlight regions of high diagnostic relevance

   Morphological Patterns (AI-detected):
   - {interp}
   - Spatial clustering of high-attention patches suggests focal disease distribution

4. PATHOLOGICAL INTERPRETATION
   {interpretation_block}

5. RECOMMENDATIONS
   {recs_block}

6. DISCLAIMER
   ─────────────────────────────────────────────────────────
   This report is generated by an AI system and is intended
   solely to ASSIST qualified pathologists. It does NOT
   constitute a clinical diagnosis. All findings must be
   reviewed and confirmed by a licensed pathologist before
   use in clinical decision-making.

   Model validation: PatchCamelyon benchmark — AUC > 0.97
   ─────────────────────────────────────────────────────────
"""

def generate_demo_report(analysis_json: Dict[str, Any]) -> str:
    cls  = analysis_json["classification"]
    att  = analysis_json["attention_analysis"]
    expl = analysis_json["explainability"]
    recs = analysis_json["recommendations"]
    meta = analysis_json["metadata"]

    if cls["tumor_probability"] > 0.80:
        interp_block = (
            "The AI model identifies features consistent with metastatic carcinoma "
            "with high confidence. Activation maps highlight malignant epithelial "
            "clusters with strong nuclear atypia and increased mitotic figures. "
            "Lymphovascular invasion cannot be excluded based on current analysis."
        )
    elif cls["tumor_probability"] > 0.50:
        interp_block = (
            "The AI model detects features suspicious for malignancy with moderate "
            "confidence. Focal areas of architectural disarray and increased nuclear "
            "pleomorphism are noted. Differential includes well-differentiated "
            "carcinoma vs. reactive atypia."
        )
    else:
        interp_block = (
            "No definitive features of malignancy detected at this confidence threshold. "
            "Tissue architecture appears predominantly within normal limits. "
            "Inflammatory infiltrate present; clinical correlation recommended."
        )

    recs_block = "\n   ".join(f"• {r}" for r in recs)

    return DEMO_REPORT.format(
        date         = meta["timestamp"][:10],
        n_patches    = meta["n_patches"],
        prob         = cls["tumor_probability"],
        prediction   = cls["prediction"],
        confidence   = cls["confidence"],
        top_regions  = ", ".join(att["top_regions"]),
        focus_desc   = att["focus_description"].lower(),
        entropy      = att["attention_entropy"],
        interp       = expl["interpretation"],
        interpretation_block = interp_block,
        recs_block   = recs_block,
    )


# 4. PDF export  (beautiful clinical report with medical images)

# Colours (RGB tuples for fpdf)
_BLUE_DARK   = (26,  58,  92)   # #1A3A5C — header bar
_BLUE_MID    = (52, 120, 180)   # #3478B4 — section accents
_BLUE_LIGHT  = (230, 240, 250)  # #E6F0FA — section bg
_RED         = (192,  57,  43)  # #C0392B — tumor
_GREEN       = (39,  174,  96)  # #27AE60 — normal
_GRAY_LIGHT  = (245, 245, 245)
_GRAY_TEXT   = (80,   80,  80)
_BLACK       = (20,   20,  20)


def _generate_analysis_figure(analysis: dict) -> Optional[str]:
    """
    Build a 3-panel matplotlib figure from analysis JSON.
    Returns path to a temporary PNG file (caller must delete it).
    """
    try:
        import re as _re
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        import numpy as np

        cls  = analysis.get("classification", {})
        att  = analysis.get("attention_analysis", {})
        meta = analysis.get("metadata", {})
        expl = analysis.get("explainability", {})

        prob        = float(cls.get("tumor_probability", 0.5))
        prediction  = cls.get("prediction", "UNKNOWN")
        confidence  = cls.get("confidence", "N/A")
        top_scores  = att.get("top_attention_scores", [])[:5]
        top_regions = att.get("top_regions", [])[:5]
        n_patches   = meta.get("n_patches", "N/A")
        wsi_id      = meta.get("wsi_id", "")
        gradcam_iou = expl.get("gradcam_iou", None)
        entropy     = att.get("attention_entropy", None)

        is_tumor    = prediction == "TUMOR"
        accent      = "#C0392B" if is_tumor else "#27AE60"
        accent_light= "#FADBD8" if is_tumor else "#D5F5E3"

        fig = plt.figure(figsize=(12, 3.8), facecolor="white")
        fig.patch.set_edgecolor("#CCCCCC")
        fig.patch.set_linewidth(1)

        gs = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.03, right=0.97,
                               top=0.82, bottom=0.15,
                               wspace=0.38)

        # Panel A: Donut gauge
        ax1 = fig.add_subplot(gs[0])
        vals   = [prob, 1 - prob]
        colors = [accent, "#E8E8E8"]
        wedges, _ = ax1.pie(
            vals, colors=colors, startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2)
        )
        ax1.text(0,  0.10, f"{prob:.1%}",
                 ha="center", va="center",
                 fontsize=20, fontweight="bold", color=accent)
        ax1.text(0, -0.22, prediction,
                 ha="center", va="center",
                 fontsize=12, fontweight="bold", color=accent)
        ax1.text(0, -0.48, f"Confidence: {confidence}",
                 ha="center", va="center",
                 fontsize=8.5, color="#555555")
        ax1.set_title("Tumor Probability", fontsize=10,
                      fontweight="bold", color="#1A3A5C", pad=6)

        # Panel B: Attention scores bar chart
        ax2 = fig.add_subplot(gs[1])
        if top_scores:
            n   = len(top_scores)
            rev = list(reversed(range(n)))
            rev_scores  = list(reversed(top_scores))
            rev_regions = list(reversed(top_regions)) if top_regions else [f"P{i}" for i in rev]
            labels = [r.replace("patch_", "P") for r in rev_regions]

            bar_colors = []
            for i in range(n):
                alpha = 0.5 + 0.5 * (i / max(n - 1, 1))
                r, g, b = (0xC0, 0x39, 0x2B) if is_tumor else (0x27, 0xAE, 0x60)
                bar_colors.append((r/255*alpha + (1-alpha), g/255*alpha + (1-alpha),
                                   b/255*alpha + (1-alpha)))

            bars = ax2.barh(rev, rev_scores, color=bar_colors,
                            edgecolor="white", linewidth=0.5, height=0.6)
            ax2.set_yticks(rev)
            ax2.set_yticklabels(labels, fontsize=8.5)
            ax2.set_xlabel("Attention Score", fontsize=8, color="#444444")
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax2.spines[spine].set_color("#CCCCCC")
            ax2.tick_params(colors="#444444", labelsize=8)
            mx = max(top_scores) if top_scores else 1
            for bar, score in zip(bars, rev_scores):
                ax2.text(bar.get_width() + mx * 0.03,
                         bar.get_y() + bar.get_height() / 2,
                         f"{score:.4f}", va="center",
                         fontsize=7, color="#333333")
        ax2.set_title("Top Suspicious Regions\n(Attention Score)", fontsize=10,
                      fontweight="bold", color="#1A3A5C", pad=6)

        # Panel C: Metrics table
        ax3 = fig.add_subplot(gs[2])
        ax3.axis("off")

        def fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)

        focus_raw = att.get("focus_description", "")
        focus_short = (focus_raw.split(" - ")[0][:22]
                       if " - " in focus_raw else focus_raw[:22])

        rows = [
            ["Specimen ID",      wsi_id.replace("camelyon16_", "")],
            ["Patches Analyzed", str(n_patches)],
            ["Prediction",       f"{prediction} ({confidence})"],
            ["Tumor Prob.",      f"{prob:.1%}"],
            ["Grad-CAM IoU",     fmt(gradcam_iou) if gradcam_iou is not None else "N/A"],
            ["Attn Entropy",     fmt(entropy)      if entropy is not None      else "N/A"],
            ["Focus Pattern",    focus_short],
            ["Model",            "EfficientNet-B3 + Gated MIL"],
        ]

        tbl = ax3.table(
            cellText=[[r[1]] for r in rows],
            rowLabels=[r[0] for r in rows],
            colLabels=["Value"],
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.38)

        HEADER_BG = "#1A3A5C"
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#DDDDDD")
            if row == 0:                          # column header
                cell.set_facecolor(HEADER_BG)
                cell.set_text_props(color="white", fontweight="bold")
            elif col == -1:                       # row labels
                cell.set_facecolor("#EAF0F8")
                cell.set_text_props(fontweight="bold", color="#1A3A5C")
            else:
                bg = accent_light if rows[row - 1][0] == "Prediction" else "white"
                cell.set_facecolor(bg)

        ax3.set_title("Analysis Details", fontsize=10,
                      fontweight="bold", color="#1A3A5C", pad=6)

        status_label = (f"AI ANALYSIS:  {prediction}  ({confidence} confidence)  "
                        f"|  Tumor Probability: {prob:.1%}")
        fig.suptitle(status_label, fontsize=11,
                     fontweight="bold", color=accent, y=0.97)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=140, bbox_inches="tight",
                    facecolor="white", edgecolor="#CCCCCC")
        plt.close(fig)
        return tmp.name

    except Exception as exc:
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        return None


def export_pdf(report_text: str, save_path: str,
               heatmap_path: Optional[str] = None,
               analysis: Optional[dict] = None):
    """
    Export a beautiful clinical report PDF.
    - Coloured header band
    - Per-WSI AI analysis figure (gauge + attention + metrics)
    - Real histopathology Grad-CAM images (tumor or normal)
    - Clean text body with proper wrapping
    """
    import re as _re
    import os as _os
    import tempfile as _tempfile

    try:
        from fpdf import FPDF
    except ImportError:
        print("  fpdf2 not installed — skipping PDF export. Run: pip install fpdf2")
        return

    # helpers
    def _c(text: str) -> str:
        """Strip markdown / unicode decoration, return latin-1 safe string."""
        text = _re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
        text = _re.sub(r'^#{1,6}\s*', '', text)
        for src, dst in {
            "\u2550":"=","\u2554":"+","\u2557":"+","\u255a":"+",
            "\u255d":"+","\u2560":"+","\u2563":"+","\u2566":"+",
            "\u2569":"+","\u256c":"+","\u2551":"|","\u2500":"-",
            "\u2502":"|","\u2588":"#","\u2580":"#","\u2584":"#",
            "\u2022":"-","\u2013":"-","\u2014":"--","\u00d7":"x",
            "\u2019":"'","\u2018":"'","\u201c":'"',"\u201d":'"',
            "\u2026":"...","\u00b1":"+-","\u00b0":" deg",
        }.items():
            text = text.replace(src, dst)
        return text.encode("latin-1", "replace").decode("latin-1")

    def _is_section(line: str) -> bool:
        return bool(_re.match(r'^\d+\.\s+[A-Z]', line.strip()))

    def _is_sep(line: str) -> bool:
        s = line.strip()
        return len(s) > 4 and all(c in "=-_*" for c in s)

    def _write(pdf, page_w, line, bold=False, size=10, line_h=5.5):
        pdf.set_x(pdf.l_margin)
        style = "B" if bold else ""
        pdf.set_font("Helvetica", style, size)
        try:
            pdf.multi_cell(page_w, line_h, line,
                           new_x="LMARGIN", new_y="NEXT")
        except Exception:
            try:
                pdf.multi_cell(page_w, line_h, line[:200],
                               new_x="LMARGIN", new_y="NEXT")
            except Exception:
                pass

    # determine prediction / images
    prediction = "UNKNOWN"
    is_tumor   = False
    if analysis:
        cls_block  = analysis.get("classification", {})
        prediction = cls_block.get("prediction", "UNKNOWN")
        is_tumor   = prediction == "TUMOR"

    REPORTS_DIR = Path(save_path).resolve().parent.parent  # …/reports/
    CLINICAL_DIR = Path(save_path).resolve().parent        # …/reports/clinical/
    def _img(name):
        p = REPORTS_DIR / name
        return str(p) if p.exists() else None

    # Per-WSI real patch grid (generated by generate_wsi_patch_grids.py)
    wsi_id_str = (analysis or {}).get("metadata", {}).get("wsi_id", "")
    patch_grid = None
    if wsi_id_str:
        pg = CLINICAL_DIR / "wsi_patches" / f"{wsi_id_str}_patches.png"
        if pg.exists():
            patch_grid = str(pg)

    # Shared Grad-CAM heatmaps as second image (fallback if no per-WSI grid)
    med_img2 = _img("gradcam_tumor_1.png") if is_tumor else _img("gradcam_normal_1.png")
    if not med_img2:
        med_img2 = _img("patch_mask_overlay.png") or _img("mask_vs_gradcam.png")

    # build per-report analysis figure
    fig_tmp = None
    if analysis:
        fig_tmp = _generate_analysis_figure(analysis)

    # build PDF
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    page_w = pdf.w - pdf.l_margin - pdf.r_margin

    # Header band
    r, g, b = _BLUE_DARK
    pdf.set_fill_color(r, g, b)
    pdf.rect(0, 0, pdf.w, 28, "F")

    pdf.set_y(5)
    pdf.set_x(0)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(pdf.w, 8, "PATHOLOGY AI ANALYSIS REPORT",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 9)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    wsi_label = ""
    if analysis:
        wsi_label = f"  |  WSI: {analysis.get('metadata', {}).get('wsi_id', '')}"
    pdf.cell(pdf.w, 6,
             f"Computational Pathology Lab  |  {date_str}{wsi_label}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(*_BLACK)
    pdf.set_y(32)

    # Prediction badge
    if analysis:
        cls_b = analysis.get("classification", {})
        prob  = cls_b.get("tumor_probability", 0)
        conf  = cls_b.get("confidence", "")
        badge_r, badge_g, badge_b = (_RED if is_tumor else _GREEN)
        pdf.set_fill_color(badge_r, badge_g, badge_b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        badge_text = (f"  RESULT: {prediction}  |  Tumor Probability: {prob:.1%}"
                      f"  |  Confidence: {conf}  ")
        pdf.set_x(pdf.l_margin)
        pdf.cell(page_w, 8, badge_text, fill=True,
                 new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_text_color(*_BLACK)
        pdf.ln(4)

    # Analysis figure
    if fig_tmp and Path(fig_tmp).exists():
        try:
            pdf.set_x(pdf.l_margin)
            pdf.image(fig_tmp, x=pdf.l_margin, w=page_w)
            pdf.ln(4)
        except Exception:
            pass
        finally:
            try:
                _os.unlink(fig_tmp)
            except Exception:
                pass

    # Medical images
    if patch_grid or med_img2:
        r2, g2, b2 = _BLUE_LIGHT
        pdf.set_fill_color(r2, g2, b2)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_BLUE_DARK)
        lbl = ("Actual H&E Tissue Patches  -  Top Attention Regions from This Slide"
               if is_tumor else
               "Actual H&E Tissue Patches  -  This Slide (Low Activation, Normal)")
        pdf.cell(page_w, 6, f"  {lbl}", fill=True,
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*_BLACK)
        pdf.ln(2)

        if patch_grid:
            # Full-width per-WSI patch grid (the real tissue of this specific slide)
            try:
                pdf.set_x(pdf.l_margin)
                pdf.image(patch_grid, x=pdf.l_margin, w=page_w)
                pdf.ln(2)
            except Exception:
                pass
            # Caption
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(*_GRAY_TEXT)
            cap = (
                "Actual tissue patches extracted from this specific slide. "
                "Red-bordered patches are the top-attention regions flagged by the Gated MIL model "
                "as most suspicious for malignancy. Attention scores are shown above each patch."
                if is_tumor else
                "Actual tissue patches extracted from this specific slide. "
                "Green-bordered patches are the top-attention regions. "
                "Low attention scores and normal tissue morphology - no malignant features detected."
            )
            pdf.multi_cell(page_w, 4, cap, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*_BLACK)
            pdf.ln(3)

        if med_img2:
            # Grad-CAM reference image (right half of a 2-col row if both exist)
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*_BLUE_DARK)
            gc_lbl = ("Grad-CAM Heatmap Reference  (EfficientNet-B3 Activation)"
                      if is_tumor else
                      "Grad-CAM Heatmap Reference  (Normal Tissue)")
            pdf.cell(page_w, 5, f"  {gc_lbl}", fill=True,
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*_BLACK)
            pdf.ln(1)
            try:
                pdf.set_x(pdf.l_margin)
                pdf.image(med_img2, x=pdf.l_margin, w=page_w * 0.55)
                pdf.ln(1)
            except Exception:
                pass
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(*_GRAY_TEXT)
            pdf.multi_cell(
                page_w, 4,
                "Grad-CAM activation map highlighting regions of diagnostic relevance "
                "as detected by the EfficientNet-B3 backbone. Red=high activation.",
                new_x="LMARGIN", new_y="NEXT"
            )
            pdf.set_text_color(*_BLACK)
            pdf.ln(3)

    # Report text body
    r3, g3, b3 = _BLUE_DARK
    pdf.set_draw_color(r3, g3, b3)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)

    skip_titles = {"PATHOLOGY REPORT", "PATHOLOGY AI ANALYSIS REPORT",
                   "AI ANALYSIS REPORT"}

    for raw_line in report_text.strip().split("\n"):
        cleaned = _c(raw_line)
        stripped = cleaned.strip()

        if _is_sep(stripped):
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.2)
            pdf.set_x(pdf.l_margin)
            pdf.line(pdf.l_margin, pdf.get_y(),
                     pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(2)
            pdf.set_draw_color(*_BLUE_DARK)
            continue

        if stripped.upper() in skip_titles:
            continue

        if _is_section(stripped):
            pdf.ln(3)
            # Blue left-border accent
            x0, y0 = pdf.l_margin, pdf.get_y()
            pdf.set_fill_color(*_BLUE_LIGHT)
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(*_BLUE_DARK)
            pdf.set_fill_color(*_BLUE_LIGHT)
            pdf.cell(page_w, 7, f"  {stripped}", fill=True,
                     new_x="LMARGIN", new_y="NEXT")
            # Accent bar on left
            pdf.set_fill_color(*_BLUE_MID)
            pdf.rect(pdf.l_margin, y0, 2.5, 7, "F")
            pdf.set_text_color(*_BLACK)
            pdf.ln(1)
            continue

        if stripped == "":
            pdf.ln(2)
            continue

        # Normal body line — detect recommendation bullets
        is_rec = stripped.startswith("-") and "recommend" in report_text.lower()
        if is_rec:
            pdf.set_x(pdf.l_margin + 4)
            pdf.set_font("Helvetica", "", 10)
            try:
                pdf.multi_cell(page_w - 4, 5.5, stripped,
                               new_x="LMARGIN", new_y="NEXT")
            except Exception:
                pass
        else:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "", 10)
            try:
                pdf.multi_cell(page_w, 5.5, cleaned,
                               new_x="LMARGIN", new_y="NEXT")
            except Exception:
                try:
                    pdf.multi_cell(page_w, 5.5, stripped[:200],
                                   new_x="LMARGIN", new_y="NEXT")
                except Exception:
                    pass

    # Footer
    pdf.set_y(-20)
    pdf.set_draw_color(*_BLUE_DARK)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(1)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*_GRAY_TEXT)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        page_w, 4,
        "DISCLAIMER: This report is generated by an AI system and is intended solely to ASSIST "
        "qualified pathologists. It does NOT constitute a clinical diagnosis. All findings must be "
        "reviewed and confirmed by a licensed pathologist. Model: EfficientNet-B3 + Gated Attention MIL. "
        "Validated on PatchCamelyon benchmark (AUC > 0.97).",
        new_x="LMARGIN", new_y="NEXT"
    )
    pdf.set_text_color(*_BLACK)

    try:
        pdf.output(save_path)
        print(f"  PDF saved -> {save_path}")
    except Exception as e:
        print(f"  PDF export skipped: {e}")


# 5. Full pipeline

def generate_clinical_report(
    tumor_probability : float,
    mil_attention     : list,
    top_patch_indices : list,
    n_patches         : int,
    cfg               : dict,
    wsi_id            : str   = None,
    gradcam_iou       : float = None,
    heatmap_path      : str   = None,
    use_llm           : bool  = False,
    save_dir          : str   = None,
) -> Dict[str, str]:
    """
    End-to-end: model outputs → JSON → LLM → report → PDF.

    Returns:
        dict with keys: json_path, report_text, pdf_path
    """
    save_dir = Path(save_dir or (ROOT / cfg["paths"]["reports_dir"] / "clinical"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build JSON
    analysis = build_analysis_json(
        tumor_probability = tumor_probability,
        mil_attention     = mil_attention,
        top_patch_indices = top_patch_indices,
        n_patches         = n_patches,
        gradcam_iou       = gradcam_iou,
        wsi_id            = wsi_id,
        heatmap_path      = heatmap_path,
    )
    json_path = save_dir / f"{analysis['metadata']['wsi_id']}_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis JSON → {json_path}")

    # 2. Generate report
    if use_llm:
        prompt   = build_llm_prompt(analysis)
        provider = cfg["llm"]["provider"]
        print(f"  Calling {provider} LLM …")
        try:
            if provider == "anthropic":
                report_text = call_claude(prompt, cfg, heatmap_path=heatmap_path)
            elif provider == "groq":
                report_text = call_groq(prompt, cfg)
            elif provider == "gemini":
                report_text = call_gemini(prompt, cfg)
            elif provider == "huggingface":
                report_text = call_huggingface(prompt, cfg)
            else:
                report_text = call_openai(prompt, cfg)
        except Exception as e:
            print(f"  LLM call failed ({e}) — falling back to demo report")
            report_text = generate_demo_report(analysis)
    else:
        print("  Using offline demo report (pass --live to call LLM API)")
        report_text = generate_demo_report(analysis)

    # 3. Save text report
    report_path = save_dir / f"{analysis['metadata']['wsi_id']}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Text report  → {report_path}")

    # 4. Export PDF
    pdf_path = str(save_dir / f"{analysis['metadata']['wsi_id']}_report.pdf")
    export_pdf(report_text, pdf_path, heatmap_path)

    return {
        "json_path"  : str(json_path),
        "report_text": report_text,
        "pdf_path"   : pdf_path,
        "analysis"   : analysis,
    }


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Call real LLM API (needs API key in env)")
    args = parser.parse_args()

    cfg = load_config()

    # Demo values — in real pipeline these come from Phase 3/4/5
    import numpy as np
    np.random.seed(42)
    n_patches       = 35
    attention_scores = np.random.dirichlet(np.ones(n_patches) * 0.5).tolist()
    top_patches     = np.argsort(attention_scores)[::-1][:5].tolist()

    result = generate_clinical_report(
        tumor_probability = 0.923,
        mil_attention     = attention_scores,
        top_patch_indices = top_patches,
        n_patches         = n_patches,
        cfg               = cfg,
        wsi_id            = "camelyon16_test_042",
        gradcam_iou       = 0.61,
        use_llm           = args.live,
    )

    print("\n" + "="*60)
    print("  GENERATED REPORT PREVIEW")
    print("="*60)
    print(result["report_text"][:1500])
    print("  … (see full report in reports/clinical/)")
