import Link from "next/link";
import { notFound } from "next/navigation";
import { aiLabel, getRun, targetName, testYears, usesConsensus, usesMarket, usesMomentum } from "@/lib/db";
import { pct, summarizeRun } from "@/lib/order";
import { Redraft } from "@/components/redraft";
import { Splits } from "@/components/splits";
import { VariantNote } from "@/components/variants";
import { WarModal } from "@/components/war-modal";

export const dynamic = "force-dynamic";

export default async function RunPage({ params, searchParams }: { params: Promise<{ id: string }>; searchParams: Promise<{ year?: string }> }) {
  const { id } = await params;
  const sp = await searchParams;
  const data = await getRun(id);
  if (!data) notFound();
  const { run, byYear, splits, variants } = data;

  const s = summarizeRun(byYear, testYears(splits));
  const years = s.years.map((y) => y.year);
  const labelled = s.years.filter((y) => y.ours != null);
  const aiWins = s.ours != null && s.nba != null && s.ours > s.nba;

  const requested = Number(sp.year);
  const year = years.includes(requested) ? requested : labelled.at(-1)?.year ?? years[0];
  const picks = byYear.get(year) ?? [];
  const acc = s.years.find((y) => y.year === year) ?? { n: 0, ours: null, nba: null };

  return (
    <main className="mx-auto w-full max-w-6xl px-6 py-16 sm:py-20">
      <Link href="/" className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground hover:text-foreground">
        ← All runs
      </Link>

      <header className="mt-8 max-w-3xl">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
          {run.tag} · {run.redraft_model}
        </p>
        <h1 className="mt-4 text-3xl sm:text-4xl font-medium tracking-[-0.03em] text-balance">
          Did the AI model draft better than NBA scouts?
        </h1>
        <p className="mt-4 text-[15px] leading-7 text-muted-foreground text-pretty">
          For each draft: what the scouts actually did on the left, what the AI model would have done on the right, and how closely each
          order matched the players&apos; real <WarModal target={run.target}>{targetName(run.target)}</WarModal> ranking. The AI model only saw earlier draft classes,
          plus player data and public mock rankings available before that draft began.
        </p>
      </header>

      <dl className="mt-12 grid grid-cols-3 gap-px overflow-hidden rounded-md border bg-border">
        {[
          { k: `${aiLabel(run.redraft_model)} accuracy`, v: pct(s.ours), dim: !aiWins },
          { k: "NBA scouts accuracy", v: pct(s.nba), dim: aiWins },
          { k: "Drafts the AI won", v: s.scored ? `${s.wins} of ${s.scored}` : "-", dim: false },
        ].map(({ k, v, dim }) => (
          <div key={k} className="bg-background px-5 py-5">
            <dt className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{k}</dt>
            <dd className={`mt-2 font-mono text-3xl tabular-nums ${dim ? "text-muted-foreground" : ""}`}>{v}</dd>
          </div>
        ))}
      </dl>
      <VariantNote v={variants} nba={s.nba} headline={usesMarket(run.redraft_model) ? "market" : usesConsensus(run.redraft_model) ? "consensus" : usesMomentum(run.redraft_model) ? "momentum" : "pure"} />

      <section className="mt-16">
        <div className="border-b pb-3">
          <h2 className="hash-heading text-lg font-medium tracking-[-0.02em]">How it was tested</h2>
        </div>
        <div className="mt-6">
          <Splits splits={splits} />
        </div>
      </section>

      <section className="mt-16">
        <div className="border-b pb-3">
          <h2 className="hash-heading text-lg font-medium tracking-[-0.02em]">Every draft</h2>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
              <th className="py-3 font-normal">Draft</th>
              <th className="py-3 text-right font-normal">{aiLabel(run.redraft_model)}</th>
              <th className="py-3 text-right font-normal">NBA scouts</th>
              <th className="py-3 text-right font-normal">Gap</th>
              <th className="py-3 pl-8 font-normal">AI minus scouts</th>
            </tr>
          </thead>
          <tbody>
            {s.years.map((y) => {
              const d = y.ours != null && y.nba != null ? y.ours - y.nba : null;
              const barWidth = d == null ? 0 : Math.min(50, 100 * Math.abs(d));
              const active = y.year === year;
              return (
                <tr key={y.year} className={`border-b transition-colors hover:bg-muted/60 ${active ? "bg-muted/40" : ""} ${y.test ? "" : "opacity-50"}`}>
                  <td className="py-2.5">
                    <Link href={`/runs/${id}?year=${y.year}`} className={`font-mono tabular-nums underline-offset-4 hover:underline ${active ? "font-medium" : ""}`}>
                      {y.year}
                    </Link>
                    {y.test && <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">test</span>}
                  </td>
                  <td className="py-2.5 text-right font-mono tabular-nums">{pct(y.ours)}</td>
                  <td className="py-2.5 text-right font-mono tabular-nums text-muted-foreground">{pct(y.nba)}</td>
                  <td className={`py-2.5 text-right font-mono tabular-nums ${d == null ? "text-muted-foreground" : d < 0 ? "text-destructive" : ""}`}>
                    {d == null ? "pending" : `${d >= 0 ? "+" : ""}${(100 * d).toFixed(0)}%`}
                  </td>
                  <td className="py-2.5 pl-8">
                    {d != null && (
                      <div className="relative h-2 w-full max-w-xs">
                        <div className="absolute inset-y-0 left-1/2 w-px bg-foreground/30" />
                        <div
                          className={`absolute inset-y-0 ${d >= 0 ? "bg-foreground" : "bg-destructive"}`}
                          style={{ [d >= 0 ? "left" : "right"]: "50%", width: `${barWidth}%` }}
                        />
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="mt-16">
        <div className="flex flex-wrap items-baseline justify-between gap-4 border-b pb-3">
          <h2 className="hash-heading text-lg font-medium tracking-[-0.02em]">{year} draft</h2>
          <nav className="flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs">
            {years.map((y) => (
              <Link key={y} href={`/runs/${id}?year=${y}`} className={y === year ? "text-foreground underline underline-offset-4" : "text-muted-foreground hover:text-foreground"}>
                {y}
              </Link>
            ))}
          </nav>
        </div>
        <div className="mt-6">
          <Redraft picks={picks} accuracy={acc} model={run.redraft_model} target={run.target} />
        </div>
      </section>
    </main>
  );
}
