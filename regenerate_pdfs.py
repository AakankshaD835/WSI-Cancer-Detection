"""
Regenerate all clinical PDFs from existing .txt + .json report files.
Reads reports/clinical/*_report.txt + *_analysis.json
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phase7_llm_report import export_pdf

clinical_dir = ROOT / "reports" / "clinical"
txt_files = sorted(clinical_dir.glob("*_report.txt"))

print(f"Found {len(txt_files)} text reports - regenerating PDFs ...\n")

ok = 0
for i, txt_path in enumerate(txt_files, 1):
    pdf_path = str(txt_path.with_suffix(".pdf"))
    report_text = txt_path.read_text(encoding="utf-8", errors="replace")

    # Load matching analysis JSON
    wsi_id = txt_path.stem.replace("_report", "")
    json_path = clinical_dir / f"{wsi_id}_analysis.json"
    analysis = None
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                analysis = json.load(f)
        except Exception:
            pass

    export_pdf(report_text, pdf_path, heatmap_path=None, analysis=analysis)
    ok += 1
    print(f"  [{i:3d}/{len(txt_files)}] {wsi_id}")

print(f"\nDone. {ok}/{len(txt_files)} PDFs regenerated in {clinical_dir}")
