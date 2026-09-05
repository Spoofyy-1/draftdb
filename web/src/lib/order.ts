import type { Pick } from "@/lib/db";

export type YearAccuracy = { year: number; holdout: boolean; n: number; ours: number | null; nba: number | null };
export type Accuracy = Omit<YearAccuracy, "year" | "holdout">;
export type RunSummary = ReturnType<typeof summarizeRun>;

/** 0.62 -> "62%". */
export const pct = (v: number | null) => (v == null ? "-" : `${(100 * v).toFixed(0)}%`);

const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null);

// Average ranks, ties share the mean rank.
function ranks(v: number[]): number[] {
  const idx = v.map((_, i) => i).sort((i, j) => v[i] - v[j]);
  const r = new Array<number>(v.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && v[idx[j + 1]] === v[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k]] = avg;
    i = j + 1;
  }
  return r;
}

/** Spearman rank correlation: 1 = identical order, 0 = unrelated, -1 = reversed. Null when undefined. */
export function spearman(a: number[], b: number[]): number | null {
  if (a.length < 3 || a.length !== b.length) return null;
  const ra = ranks(a), rb = ranks(b);
  const ma = mean(ra) as number, mb = mean(rb) as number;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < ra.length; i++) {
    num += (ra[i] - ma) * (rb[i] - mb);
    da += (ra[i] - ma) ** 2;
    db += (rb[i] - mb) ** 2;
  }
  return da && db ? num / Math.sqrt(da * db) : null;
}

/** How well each draft order (the AI model's, the scouts') ranked one class by the realised WAR target, over the same players. */
export function orderAccuracy(picks: Pick[]): Accuracy {
  const rows = picks.filter((p) => p.modelled && p.labelled && p.war != null && p.pred != null);
  const war = rows.map((p) => p.war as number);
  return {
    n: rows.length,
    ours: spearman(rows.map((p) => p.pred as number), war),
    nba: spearman(rows.map((p) => -p.actual_pick), war), // pick 1 is the best slot, so negate
  };
}

/** Per-year accuracy for every redrafted year, plus run-level averages over the holdout years only. */
export function summarizeRun(byYear: Map<number, Pick[]>, holdoutYears: number[]) {
  const years: YearAccuracy[] = [...byYear.keys()]
    .sort((a, b) => a - b)
    .map((year) => ({ year, holdout: holdoutYears.includes(year), ...orderAccuracy(byYear.get(year) ?? []) }));
  const scored = years.filter((y) => y.holdout && y.ours != null && y.nba != null);
  return {
    years,
    scored: scored.length,
    ours: mean(scored.map((y) => y.ours as number)),
    nba: mean(scored.map((y) => y.nba as number)),
    wins: scored.filter((y) => (y.ours as number) > (y.nba as number)).length,
  };
}
