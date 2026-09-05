import path from "node:path";
import fs from "node:fs";
import type { Database as BunDatabase } from "bun:sqlite";
import { summarizeRun, type RunSummary } from "@/lib/order";

// Next runs under the Bun runtime (`bun --bun next dev`); bun:sqlite is built in, no native addon needed.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { Database } = require(/* turbopackIgnore: true */ "bun:sqlite") as typeof import("bun:sqlite");

const DB_PATH = process.env.RESULTS_DB ?? path.resolve(process.cwd(), "..", "outputs", "results.sqlite");

function open() {
  if (!fs.existsSync(DB_PATH)) return null;
  return new Database(DB_PATH, { readonly: true }) as BunDatabase;
}

export type Run = {
  run_id: string; created: string; tag: string; models: string; redraft_model: string; north_star: string; target: TargetKind; target_desc: string;
  feature_hash: string; n_features: number; holdout_evaluated: number; gpus: number;
};
/** The score both draft orders are judged against. "war3": WAR summed over a player's first three NBA seasons;
 * "peak": mean WAR over his three best seasons. */
export type TargetKind = "war3" | "peak";
export const targetName = (t: TargetKind) => (t === "peak" ? "peak WAR" : "3-year WAR");
/** One row of one redraft: the real pick plus, for players the model could score, its prediction and re-ranked slot.
 * `war` is the run's target for that player (null until he has played a season). */
export type Pick = {
  year: number; model: string; protocol: string | null; player: string; bbref_id: string; actual_pick: number; new_pick: number | null;
  team: string; college: string | null; modelled: number; pred: number | null; war: number | null; seasons_played: number; labelled: number;
};

/** What one draft class was used for in a run, and which earlier classes trained the model that scored it. */
export type Split = { year: number; role: "no_features" | "context" | "validation" | "holdout" | "unlabelled"; labelled: number; causal_context: number[] };

const PICKS_SQL = "SELECT * FROM redraft_picks WHERE run_id=? ORDER BY year, actual_pick";

function groupByYear(rows: Pick[]): Map<number, Pick[]> {
  const out = new Map<number, Pick[]>();
  for (const r of rows) out.set(r.year, [...(out.get(r.year) ?? []), r]);
  return out;
}

function readSplits(db: BunDatabase, id: string): Split[] {
  const rows = db.query("SELECT year, role, labelled, causal_context FROM splits WHERE run_id=? ORDER BY year").all(id) as (Omit<Split, "causal_context"> & { causal_context: string })[];
  return rows.map((r) => ({ ...r, causal_context: JSON.parse(r.causal_context) as number[] }));
}

/** The draft classes a run is judged on. */
export const testYears = (splits: Split[]) => splits.filter((s) => s.role === "validation" || s.role === "holdout").map((s) => s.year);

/** Explicitly labelled variants of the pure redraft model. */
export const usesMarket = (model: string) => model.endsWith("+market");
export const usesConsensus = (model: string) => model.endsWith("+consensus");
export const usesMomentum = (model: string) => model.includes("+momentum");
export const aiLabel = (model: string) => (usesMarket(model) ? "AI + scouts' pick" : "AI model");
export const pureModel = (model: string) => model.replace("+market", "").replace("+consensus", "").replace("+momentum", "");

/** Mean test-year Spearman of each variant of the redraft's base model present in the run's metrics. */
export type Variants = { pure: number | null; momentum: number | null; consensus: number | null; market: number | null };
function variantAccuracies(db: BunDatabase, runId: string, model: string): Variants {
  const base = pureModel(model);
  const q = db.query("SELECT AVG(spearman) AS s FROM metrics WHERE run_id=? AND protocol='causal' AND model=?");
  const get = (m: string) => (q.get(runId, m) as { s: number | null } | undefined)?.s ?? null;
  return { pure: get(base), momentum: get(`${base}+momentum`), consensus: get(`${base}+consensus`), market: get(`${base}+market`) };
}

/** Runs, newest first, each with its draft-order accuracy summary (AI model vs. NBA scouts) over its test years. */
export function listRuns(): (Run & RunSummary & { variants: Variants })[] {
  const db = open();
  if (!db) return [];
  const runs = db.query("SELECT * FROM runs ORDER BY created DESC").all() as Run[];
  const picks = db.query(PICKS_SQL);
  const out = runs.map((r) => ({
    ...r,
    ...summarizeRun(groupByYear(picks.all(r.run_id) as Pick[]), testYears(readSplits(db, r.run_id))),
    variants: variantAccuracies(db, r.run_id, r.redraft_model),
  }));
  db.close();
  return out;
}

/** The split table of one run: what each draft class was used for. */
export function getSplits(id: string): Split[] {
  const db = open();
  if (!db) return [];
  const splits = readSplits(db, id);
  db.close();
  return splits;
}

/** One run plus every redraft pick in it, grouped by draft year. Null if the run does not exist. */
export function getRun(id: string): { run: Run; byYear: Map<number, Pick[]>; splits: Split[]; variants: Variants } | null {
  const db = open();
  if (!db) return null;
  const run = db.query("SELECT * FROM runs WHERE run_id=?").get(id) as Run | undefined;
  const picks = run ? (db.query(PICKS_SQL).all(id) as Pick[]) : [];
  const splits = run ? readSplits(db, id) : [];
  const variants = run ? variantAccuracies(db, id, run.redraft_model) : { pure: null, momentum: null, consensus: null, market: null };
  db.close();
  return run ? { run, byYear: groupByYear(picks), splits, variants } : null;
}
