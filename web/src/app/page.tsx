import Link from "next/link";
import { listRuns } from "@/lib/db";
import { pct } from "@/lib/order";
import { WarModal } from "@/components/war-modal";

export const dynamic = "force-dynamic";

export default function Home() {
  const runs = listRuns();
  const latest = runs[0];
  const aiWins = latest?.ours != null && latest?.nba != null && latest.ours > latest.nba;

  return (
    <main className="mx-auto w-full max-w-4xl px-6 py-16 sm:py-24">
      <header className="max-w-2xl">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">NBA Draft · AI redraft</p>
        <h1 className="mt-5 text-4xl sm:text-5xl font-medium tracking-[-0.035em] text-balance">
          Did the AI model draft better than NBA scouts?
        </h1>
        <p className="mt-5 text-[15px] leading-7 text-muted-foreground text-pretty">
          We re-ran every draft since 2010 with an AI model that only knew what was knowable on draft night. Then we checked
          whose order better matched how the players actually turned out: the AI model, or the NBA scouts who made the real
          picks. Players are ranked by <WarModal>peak WAR</WarModal>, how many wins a season they were worth at their best.
        </p>
      </header>

      {latest && (
        <dl className="mt-14 grid grid-cols-3 gap-px overflow-hidden rounded-md border bg-border">
          {[
            { k: "AI model accuracy", v: pct(latest.ours), dim: !aiWins },
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
        Order accuracy: how closely a draft order matches the players&apos; actual peak WAR ranking. 100% is a perfect order,
        0% is no relationship.
      </p>

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
                <th className="py-3 text-right font-normal">AI model</th>
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
                  <td className="py-3.5 pr-4 font-mono text-xs text-muted-foreground">{r.redraft_model}</td>
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
