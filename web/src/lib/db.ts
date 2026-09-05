import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { summarizeRun, type RunSummary } from "@/lib/order";

// Results live in Supabase Postgres. Reads use the publishable key against the tables' anon select policy;
// every query is async and PostgREST-paged, so nothing blocks the render on a single large fetch.
const URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const KEY = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
const PAGE = 1000; // PostgREST's default max rows per request

function open(): SupabaseClient | null {
  if (!URL || !KEY) return null;
  return createClient(URL, KEY, { auth: { persistSession: false } });
}

export type Run = {
  run_id: string; created: string; tag: string; models: string; redraft_model: string; north_star: string; target: TargetKind; target_desc: string;
  feature_hash: string; n_features: number; gpus: number;
};
/** The score both draft orders are judged against. "war5": WAR summed over a player's first five NBA seasons;
 * "peak": mean WAR over his five best seasons. */
export type TargetKind = "war5" | "peak";
export const targetName = (t: TargetKind) => (t === "peak" ? "peak WAR" : "5-year WAR");
/** One row of one redraft: the real pick plus, for players the model could score, its prediction and re-ranked slot.
 * `war` is the run's target for that player (null until he has played a season). */
export type Pick = {
  year: number; model: string; protocol: string | null; player: string; bbref_id: string; actual_pick: number; new_pick: number | null;
  team: string; college: string | null; modelled: number; pred: number | null; war: number | null; seasons_played: number; labelled: number;
};

/** What one draft class was used for in a run, and which earlier classes trained the model that scored it. */
export type Split = { year: number; role: "no_features" | "context" | "holdout" | "unlabelled"; labelled: number; causal_context: number[] };

/** Read a whole table selection, following PostgREST's row cap until a short page comes back. */
async function page<T>(db: SupabaseClient, table: string, runId: string, order: string[]): Promise<T[]> {
  const out: T[] = [];
  for (let from = 0; ; from += PAGE) {
    let q = db.from(table).select("*").eq("run_id", runId);
    for (const col of order) q = q.order(col);
    const { data, error } = await q.range(from, from + PAGE - 1);
    if (error) throw new Error(`${table}: ${error.message}`);
    const rows = (data ?? []) as T[];
    out.push(...rows);
    if (rows.length < PAGE) return out;
  }
}

function groupByYear(rows: Pick[]): Map<number, Pick[]> {
  const out = new Map<number, Pick[]>();
  for (const r of rows) out.set(r.year, [...(out.get(r.year) ?? []), r]);
  return out;
}

const readPicks = (db: SupabaseClient, id: string) => page<Pick>(db, "redraft_picks", id, ["year", "actual_pick"]);
const readSplits = (db: SupabaseClient, id: string) => page<Split>(db, "splits", id, ["year"]);

/** The draft classes a run is judged on. */
export const holdoutYears = (splits: Split[]) => splits.filter((s) => s.role === "holdout").map((s) => s.year);

/** Explicitly labelled variants of the pure redraft model. */
export const usesMarket = (model: string) => model.endsWith("+market");
export const usesConsensus = (model: string) => model.endsWith("+consensus");
export const usesMomentum = (model: string) => model.includes("+momentum");
export const aiLabel = (model: string) => (usesMarket(model) ? "AI + scouts' pick" : "AI model");
export const pureModel = (model: string) => model.replace("+market", "").replace("+consensus", "").replace("+momentum", "");

/** Mean holdout-year Spearman of each variant of the redraft's base model present in the run's metrics. */
export type Variants = { pure: number | null; momentum: number | null; consensus: number | null; market: number | null };
type MetricRow = { model: string; spearman: number | null };

/** PostgREST has no AVG, so the run's causal metric rows are averaged here -- a few hundred rows per run. */
async function variantAccuracies(db: SupabaseClient, runId: string, model: string): Promise<Variants> {
  const { data, error } = await db.from("metrics").select("model, spearman").eq("run_id", runId).eq("protocol", "causal");
  if (error) throw new Error(`metrics: ${error.message}`);
  const rows = (data ?? []) as MetricRow[];
  const base = pureModel(model);
  const get = (m: string) => {
    const vals = rows.filter((r) => r.model === m && r.spearman != null).map((r) => r.spearman as number);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };
  return { pure: get(base), momentum: get(`${base}+momentum`), consensus: get(`${base}+consensus`), market: get(`${base}+market`) };
}

/** Runs, newest first, each with its draft-order accuracy summary (AI model vs. NBA scouts) over its holdout years. */
export async function listRuns(): Promise<(Run & RunSummary & { variants: Variants })[]> {
  const db = open();
  if (!db) return [];
  const { data, error } = await db.from("runs").select("*").order("created", { ascending: false });
  if (error) throw new Error(`runs: ${error.message}`);
  const runs = (data ?? []) as Run[];
  return Promise.all(
    runs.map(async (r) => {
      const [picks, splits, variants] = await Promise.all([readPicks(db, r.run_id), readSplits(db, r.run_id), variantAccuracies(db, r.run_id, r.redraft_model)]);
      return { ...r, ...summarizeRun(groupByYear(picks), holdoutYears(splits)), variants };
    }),
  );
}

/** The split table of one run: what each draft class was used for. */
export async function getSplits(id: string): Promise<Split[]> {
  const db = open();
  return db ? readSplits(db, id) : [];
}

/** One run plus every redraft pick in it, grouped by draft year. Null if the run does not exist. */
export async function getRun(id: string): Promise<{ run: Run; byYear: Map<number, Pick[]>; splits: Split[]; variants: Variants } | null> {
  const db = open();
  if (!db) return null;
  const { data, error } = await db.from("runs").select("*").eq("run_id", id).maybeSingle();
  if (error) throw new Error(`runs: ${error.message}`);
  if (!data) return null;
  const run = data as Run;
  const [picks, splits, variants] = await Promise.all([readPicks(db, id), readSplits(db, id), variantAccuracies(db, id, run.redraft_model)]);
  return { run, byYear: groupByYear(picks), splits, variants };
}
