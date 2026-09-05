import { Fragment } from "react";
import { targetName, type Pick, type TargetKind } from "@/lib/db";
import { pct, type Accuracy } from "@/lib/order";

const ROUND_1_PICKS = 30;
// A value that reads as a clear hit: 5 WAR a season at his peak, or 10 WAR over his first three seasons.
const STAR_WAR: Record<TargetKind, number> = { peak: 5, war3: 10 };

// Never-played players carry a sentinel value below every real one; show the fact, not the number.
const war = (p: Pick) => (p.war == null ? "-" : p.seasons_played === 0 ? "never played" : p.war.toFixed(1));

type Row = Pick & { slot: number };

function Column({ title, subtitle, rows, showRealPick, target }: { title: string; subtitle: string; rows: Row[]; showRealPick?: boolean; target: TargetKind }) {
  const star = STAR_WAR[target] ?? STAR_WAR.war3;
  const label = "font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground";
  return (
    <div>
      <div className="flex items-baseline justify-between border-b pb-2">
        <h3 className="text-sm font-medium">{title}</h3>
        <span className={label}>{subtitle}</span>
      </div>
      <table className="w-full text-[13px]">
        <thead>
          <tr className={`border-b text-left ${label}`}>
            <th className="py-2 font-normal">#</th>
            <th className="py-2 font-normal">Player</th>
            {showRealPick && <th className="py-2 text-right font-normal">Real</th>}
            <th className="py-2 text-right font-normal">Seasons</th>
            <th className="py-2 text-right font-normal">{targetName(target)}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <Fragment key={p.bbref_id}>
              {p.slot === ROUND_1_PICKS + 1 && (
                <tr>
                  <td colSpan={showRealPick ? 5 : 4} className={`pt-5 pb-1.5 ${label}`}>
                    Round 2
                  </td>
                </tr>
              )}
              <tr className={`border-b border-border/60 ${p.modelled ? "" : "text-muted-foreground/70"}`}>
                <td className="w-8 py-1.5 font-mono text-xs text-muted-foreground tabular-nums">{p.slot}</td>
                <td className="py-1.5 pr-3">
                  <span className={p.modelled ? "" : "line-through decoration-muted-foreground/40"}>{p.player}</span>
                  {p.college && <span className="ml-2 text-xs text-muted-foreground">{p.college}</span>}
                  {!p.modelled && <span className="ml-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">no pre-draft data</span>}
                </td>
                {showRealPick && <td className="w-12 py-1.5 text-right font-mono text-xs text-muted-foreground tabular-nums">{p.actual_pick}</td>}
                <td className="w-14 py-1.5 text-right font-mono text-xs text-muted-foreground tabular-nums">{p.labelled ? p.seasons_played : ""}</td>
                <td
                  className={`w-24 py-1.5 text-right font-mono text-xs tabular-nums ${
                    p.war == null ? "" : p.war >= star ? "font-semibold" : p.war <= 0 ? "text-muted-foreground" : ""
                  }`}
                >
                  {war(p)}
                </td>
              </tr>
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One draft class: accuracy strip, then the real order beside the AI model's order. */
export function Redraft({ picks, accuracy, model, target }: { picks: Pick[]; accuracy: Accuracy; model: string; target: TargetKind }) {
  const market = model.endsWith("+market");
  if (!picks.length) return <p className="text-sm text-muted-foreground">No picks.</p>;
  const real = [...picks].sort((a, b) => a.actual_pick - b.actual_pick).map((p, i) => ({ ...p, slot: i + 1 }));
  const ai = picks
    .filter((p) => p.new_pick != null)
    .sort((a, b) => (a.new_pick as number) - (b.new_pick as number))
    .map((p, i) => ({ ...p, slot: i + 1 }));
  const scored = accuracy.ours != null && accuracy.nba != null;
  const aiWins = scored && (accuracy.ours as number) > (accuracy.nba as number);

  return (
    <div className="space-y-8">
      <dl className="grid grid-cols-3 gap-px overflow-hidden rounded-md border bg-border">
        {[
          { k: `${market ? "AI + scouts' pick" : "AI model"} accuracy`, v: pct(accuracy.ours), dim: !aiWins },
          { k: "NBA scouts accuracy", v: pct(accuracy.nba), dim: aiWins },
          { k: "Players compared", v: scored ? `${accuracy.n} of ${picks.length}` : "No seasons played yet", dim: false },
        ].map(({ k, v, dim }) => (
          <div key={k} className="bg-background px-5 py-4">
            <dt className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{k}</dt>
            <dd className={`mt-1.5 font-mono text-xl tabular-nums ${dim ? "text-muted-foreground" : ""}`}>{v}</dd>
          </div>
        ))}
      </dl>
      <div className="grid gap-10 lg:grid-cols-2">
        <Column title="What NBA scouts did" subtitle="Real draft order" rows={real} target={target} />
        <Column
          title={market ? "What AI + scouts would have done" : "What the AI model would have done"}
          subtitle={market ? "AI ranking blended 60 / 40 with the real pick order" : `Ranked by predicted ${targetName(target)}`}
          rows={ai}
          showRealPick
          target={target}
        />
      </div>
    </div>
  );
}
