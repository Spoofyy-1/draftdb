"""Regenerate README.md from an archived winner: headline, by-draft table and the full side-by-side redraft per class.

Usage: python -m tournament.readme --name <winners/<name>> [--models "<markdown rows for the AI models table>"]
"""

import argparse
import json

import numpy as np
import pandas as pd

from infra import config as C
from tournament.layer2 import _recompute, wide
from tournament.winner import WINNERS


def _final_score(rec: dict, l1: pd.DataFrame) -> pd.DataFrame:
    l1 = l1[l1.years.astype(str).str.split(":").str[0] == rec["window"]].assign(years=rec["window"])
    w = wide(l1, rec["window"])
    configs = sorted(l1.config.unique())
    w["_final"] = _recompute(w, configs, rec["layer2_rule"]) if rec.get("layer2_rule") else w[rec["layer1_config_name"]]
    return w


def render(name: str, models_md: str) -> str:
    rec = json.loads((WINNERS / name / "winner.json").read_text())
    l1 = pd.read_parquet(WINNERS / name / "predictions.parquet")
    w = _final_score(rec, l1)
    m = rec["metrics"]
    pct = lambda v: f"{100 * v:.0f}%"
    out = ["# nba-redraft", "", "## AI models used", "", models_md, "",
           f"## Holdout ({min(rec['years'])}-{max(rec['years'])})", "",
           "| AI model accuracy | NBA scouts accuracy | Drafts the AI won |", "|:-----------------:|:-------------------:|:-----------------:|",
           f"| **{pct(m['spearman'])}**           | {pct(m['spearman_nba'])}                 | **{m['wins']} of {m['n_years']}**        |", "",
           "Accuracy = Spearman rank correlation between the order (AI redraft, or the real draft) and the players' realised 5-year WAR",
           f"(classes that have played fewer than five seasons are judged on the seasons they have played). Rule: `{rec.get('layer2_rule') or rec['layer1_config_name']}`.", "",
           "## By draft", "", "| Draft | AI model | NBA scouts | Difference |", "|:------|---------:|-----------:|-----------:|"]
    for y, r in sorted(m["per_year"].items()):
        out.append(f"| {y}  | {pct(r['spearman'])}      | {pct(r['spearman_nba'])}        | {100 * (r['spearman'] - r['spearman_nba']):+.0f}         |")
    out += [f"| **Mean** | **{pct(m['spearman'])}** | **{pct(m['spearman_nba'])}**   | **{100 * (m['spearman'] - m['spearman_nba']):+.0f}**    |", "",
            "## By player", "", "Every holdout draft, actual order beside the model's reordering of the same names.", ""]
    for y, g in w.groupby("draft_year"):
        g = g.copy()
        actual = g.sort_values("pick")
        model = g.sort_values("_final", ascending=False)
        out += [f"### {int(y)}", "", "| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |", "|--:|:-------------|--------:|:--------------|--------:|"]
        for i, (a, b) in enumerate(zip(actual.itertuples(), model.itertuples()), 1):
            out.append(f"| {i} | {a.player} | {getattr(a, C.TARGET):.1f} | {b.player} | {getattr(b, C.TARGET):.1f} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", required=True, help="markdown table rows describing the AI models")
    a = ap.parse_args()
    text = render(a.name, a.models)
    (C.ROOT / "README.md").write_text(text)
    print(f"README.md written from winners/{a.name} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
