from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

HTML_TMPL = """
<!doctype html>
<html><head><meta charset="utf-8"><title>FraudFlow — Report</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:24px;}}
 h1{{margin:0 0 8px 0}}
 .kpi{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin:16px 0}}
 .card{{border:1px solid #eee;border-radius:12px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,.05)}}
 table{{border-collapse:collapse;width:100%;}}
 th,td{{border-bottom:1px solid #eee;padding:8px;text-align:right}}
 th:first-child, td:first-child{{text-align:left}}
 figure{{margin:0 0 16px 0}}
 .muted{{color:#777}}
</style></head><body>
<h1>FraudFlow — Model Report</h1>
<p class="muted">Gold Parquet: {gold_path}</p>
<div class="kpi">
  <div class="card"><div>ROC-AUC</div><div style="font-size:24px">{roc:.4f}</div></div>
  <div class="card"><div>PR-AUC</div><div style="font-size:24px">{pr:.4f}</div></div>
  <div class="card"><div>P@0.1%</div><div style="font-size:24px">{p001:.3f}</div></div>
  <div class="card"><div>R@0.1%</div><div style="font-size:24px">{r001:.3f}</div></div>
  <div class="card"><div>P@0.5%</div><div style="font-size:24px">{p005:.3f}</div></div>
  <div class="card"><div>R@0.5%</div><div style="font-size:24px">{r005:.3f}</div></div>
  <div class="card"><div>P@1%</div><div style="font-size:24px">{p01:.3f}</div></div>
  <div class="card"><div>R@1%</div><div style="font-size:24px">{r01:.3f}</div></div>
</div>
<figure>
  <img src="{pr_path}" alt="PR curve" style="width:100%;max-width:960px" />
  <figcaption>Precision-Recall curve</figcaption>
</figure>
<figure>
  <img src="{roc_path}" alt="ROC curve" style="width:100%;max-width:960px" />
  <figcaption>ROC curve</figcaption>
</figure>
<table>
  <thead><tr><th>step</th><th>amount</th><th>score</th></tr></thead>
  <tbody>
    {top_rows}
  </tbody>
</table>
</body></html>
"""


def pct_k(n, k):
    return max(int(np.ceil(n * k)), 1)

def precision_recall_at_k(y_true, y_score, k):
    n = len(y_score)
    top = np.argsort(-y_score)[:pct_k(n, k)]
    tp = y_true[top].sum()
    prec = tp / len(top)
    rec = tp / y_true.sum() if y_true.sum() > 0 else 0.0
    return float(prec), float(rec)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold/scores.parquet")
    ap.add_argument("--out", default="reports/metrics.html")
    args = ap.parse_args()
    gold = Path(args.gold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(gold)
    assert {"score","isFraud"}.issubset(df.columns)
    y = df["isFraud"].to_numpy()
    s = df["score"].to_numpy()

    # ROC / PR points
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, s)
    pr_auc = average_precision_score(y, s)

    # Curves
    roc_path = out.with_suffix(".roc.png")
    pr_path = out.with_suffix(".pr.png")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend()
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"AP = {pr_auc:.4f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision‑Recall'); plt.legend()
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()

    # P@k / R@k
    p001, r001 = precision_recall_at_k(y, s, 0.001)
    p005, r005 = precision_recall_at_k(y, s, 0.005)
    p01, r01   = precision_recall_at_k(y, s, 0.01)

    # Top rows HTML
    top = df.sort_values("score", ascending=False)[[c for c in ["step","amount","score"] if c in df.columns]].head(50)
    top_html = "\n".join(
        f"<tr><td>{int(r.step) if 'step' in top.columns else ''}</td><td>{float(r.amount):,.2f}</td><td>{r.score:.6f}</td></tr>"
        for r in top.itertuples(index=False)
    )

    html = HTML_TMPL.format(
        gold_path=str(gold), roc=roc_auc, pr=pr_auc,
        p001=p001, r001=r001, p005=p005, r005=r005, p01=p01, r01=r01,
        pr_path=pr_path.name, roc_path=roc_path.name,
        top_rows=top_html,
    )

    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")