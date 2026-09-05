import type { Variants } from "@/lib/db";
import { pct } from "@/lib/order";

/** Every order a run can produce, side by side, so nobody mistakes public scouting information for pure AI. */
export function VariantNote({ v, nba, headline }: { v: Variants; nba: number | null; headline: string }) {
  const rows = [
    { k: "pure", label: "AI alone", v: v.pure, note: "never sees the pick or anyone's mock draft" },
    { k: "momentum", label: "AI + pre-draft mock history", v: v.momentum, note: "uses dated mock ranks and their movement through draft night — public pre-draft information, not pure AI" },
    { k: "consensus", label: "AI + pre-draft consensus", v: v.consensus, note: "blended 60 / 40 with the average of mock drafts published before draft night — everything still pre-draft" },
    { k: "market", label: "AI + scouts' actual pick", v: v.market, note: "blended 60 / 40 with the real draft order — uses the answer key, so not a pre-draft result" },
  ].filter((r) => r.v != null);
  if (rows.length < 2) return null;
  return (
    <dl className="mt-5 grid gap-x-8 gap-y-2 text-[13px] leading-6 sm:grid-cols-2 lg:grid-cols-4">
      {rows.map((r) => (
        <div key={r.k} className={r.k === headline ? "" : "text-muted-foreground"}>
          <dt className="font-mono text-[10px] uppercase tracking-[0.16em]">{r.label}{r.k === headline ? " · shown above" : ""}</dt>
          <dd>
            <span className={`font-mono tabular-nums ${r.v != null && nba != null && r.v > nba ? "text-foreground" : ""}`}>{pct(r.v)}</span>
            <span className="text-muted-foreground"> vs scouts {pct(nba)} · {r.note}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}
