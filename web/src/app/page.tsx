import Link from "next/link";
import { aiLabel, getSplits, listRuns, targetName, usesConsensus, usesMarket, usesMomentum } from "@/lib/db";
import { pct } from "@/lib/order";
import { Splits } from "@/components/splits";
import { VariantNote } from "@/components/variants";
import { WarModal } from "@/components/war-modal";

export const dynamic = "force-dynamic";

export default async function Home() {
  const runs = await listRuns();
  const latest = runs[0];
  const aiWins = latest?.ours != null && latest?.nba != null && latest.ours > latest.nba;
  const splits = latest ? await getSplits(latest.run_id) : [];
  const target = latest?.target ?? "war5";

  return (
    <main className="mx-auto w-full max-w-4xl px-6 py-16 sm:py-24">
      <header className="max-w-2xl">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">NBA Draft · AI redraft</p>
        <h1 className="mt-5 text-4xl sm:text-5xl font-medium tracking-[-0.035em] text-balance">
          Did the AI model draft better than NBA scouts?
        </h1>
        <p className="mt-5 text-[15px] leading-7 text-muted-foreground text-pretty">
          We trained on the 2003–2018 drafts, then re-ran 2019–2025 using only information available before each draft—including
          player data and public mock rankings. Then we checked
          whose order better matched how the players actually turned out: the AI model, or the NBA scouts who made the real
          picks. Players are ranked by <WarModal target={target}>{targetName(target)}</WarModal>,{" "}
          {target === "peak" ? "how many wins a season they were worth at their best." : "how many wins they were worth over their first five NBA seasons."}
        </p>
      </header>

      {latest && (
        <dl className="mt-14 grid grid-cols-3 gap-px overflow-hidden rounded-md border bg-border">
          {[
            { k: `${aiLabel(latest.redraft_model)} accuracy`, v: pct(latest.ours), dim: !aiWins },
            { k: "NBA scouts accuracy", v: pct(latest.nba), dim: aiWins },
            { k: "Drafts the AI won", v: latest.scored ? `${latest.wins} of ${latest.scored}` : "-", dim: false },
          ].map(({ k, v, dim }) => (
            <div key={k} className="bg-background px-5 py-5">
              <dt className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{k}</dt>
              <dd className={`mt-2 font-mono text-3xl tabular-nums ${dim ? "text-muted-foreground" : ""}`}>{v}</dd>
            </div>
          ))}
        </dl>
      )}

      <p className="mt-5 text-[13px] leading-6 text-muted-foreground">
        Order accuracy: how closely a draft order matches the players&apos; actual {targetName(target)} ranking. 100% is a perfect order,
        0% is no relationship. Averaged over the holdout drafts only.
      </p>
      {latest && <VariantNote v={latest.variants} nba={latest.nba} headline={usesMarket(latest.redraft_model) ? "market" : usesConsensus(latest.redraft_model) ? "consensus" : usesMomentum(latest.redraft_model) ? "momentum" : "pure"} />}

      {splits.length > 0 && (
        <section className="mt-20">
          <div className="border-b pb-3">
            <h2 className="hash-heading text-lg font-medium tracking-[-0.02em]">How it was scored</h2>
          </div>
          <div className="mt-6">
            <Splits splits={splits} />
          </div>
        </section>
      )}

      <section className="mt-20">
        <div className="border-b pb-3">
          <h2 className="hash-heading text-lg font-medium tracking-[-0.02em]">Runs</h2>
        </div>

        {runs.length === 0 ? (
          <div className="border-b py-16 text-center">
            <p className="text-sm text-muted-foreground">No runs yet. The first one is computing.</p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                <th className="py-3 font-normal">When</th>
                <th className="py-3 font-normal">Model</th>
                <th className="py-3 text-right font-normal">AI</th>
                <th className="py-3 text-right font-normal">NBA scouts</th>
                <th className="py-3 text-right font-normal">Drafts the AI won</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className="group border-b transition-colors hover:bg-muted/60">
                  <td className="py-3.5 pr-4">
                    <Link href={`/runs/${r.run_id}`} className="tabular-nums underline-offset-4 group-hover:underline">
                      {r.created.slice(0, 16).replace("T", " ")}
                    </Link>
                  </td>
                  <td className="py-3.5 pr-4 font-mono text-xs text-muted-foreground">
                    {r.redraft_model}
                    {usesMarket(r.redraft_model) && <span className="ml-2 text-[10px] uppercase tracking-[0.16em]">sees pick</span>}
                    {usesConsensus(r.redraft_model) && <span className="ml-2 text-[10px] uppercase tracking-[0.16em]">pre-draft consensus</span>}
                  </td>
                  <td className="py-3.5 pr-4 text-right font-mono tabular-nums">{pct(r.ours)}</td>
                  <td className="py-3.5 pr-4 text-right font-mono tabular-nums text-muted-foreground">{pct(r.nba)}</td>
                  <td className="py-3.5 text-right font-mono tabular-nums">{r.scored ? `${r.wins} of ${r.scored}` : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
