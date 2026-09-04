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
  run_id: string; created: string; tag: string; models: string; redraft_model: string; north_star: string; target: string;
  feature_hash: string; n_features: number; holdout_evaluated: number; gpus: number;
};
/** One row of one redraft: the real pick plus, for players the model could score, its prediction and re-ranked slot. */
export type Pick = {
  year: number; model: string; protocol: string | null; player: string; bbref_id: string; actual_pick: number; new_pick: number | null;
  team: string; college: string | null; modelled: number; pred: number | null; peak_war: number | null; seasons_played: number; labelled: number;
};

const PICKS_SQL = "SELECT * FROM redraft_picks WHERE run_id=? ORDER BY year, actual_pick";

function groupByYear(rows: Pick[]): Map<number, Pick[]> {
  const out = new Map<number, Pick[]>();
  for (const r of rows) out.set(r.year, [...(out.get(r.year) ?? []), r]);
  return out;
}

/** Runs, newest first, each with its draft-order accuracy summary (AI model vs. NBA scouts). */
export function listRuns(): (Run & RunSummary)[] {
  const db = open();
  if (!db) return [];
  const runs = db.query("SELECT * FROM runs ORDER BY created DESC").all() as Run[];
  const picks = db.query(PICKS_SQL);
  const out = runs.map((r) => ({ ...r, ...summarizeRun(groupByYear(picks.all(r.run_id) as Pick[])) }));
  db.close();
  return out;
}

/** One run plus every redraft pick in it, grouped by draft year. Null if the run does not exist. */
export function getRun(id: string): { run: Run; byYear: Map<number, Pick[]> } | null {
  const db = open();
  if (!db) return null;
  const run = db.query("SELECT * FROM runs WHERE run_id=?").get(id) as Run | undefined;
  const picks = run ? (db.query(PICKS_SQL).all(id) as Pick[]) : [];
  db.close();
  return run ? { run, byYear: groupByYear(picks) } : null;
}
