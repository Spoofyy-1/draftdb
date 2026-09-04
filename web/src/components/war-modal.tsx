"use client";

import { useRef, type ReactNode } from "react";

const INPUTS: [string, string][] = [
  ["rating", "How many points per 100 possessions the player is worth compared with an average player (FiveThirtyEight RAPTOR, or BPM on the same scale after 2022)."],
  ["2.75", "Replacement level. A bench-level player is about 2.75 points per 100 worse than average and earns zero WAR."],
  ["minutes", "Regular-season minutes played that season."],
  ["0.000514", "Converts points and minutes into wins."],
];

/** Inline link that opens a modal explaining the peak WAR score both draft orders are judged against. */
export function WarModal({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  return (
    <>
      <button
        type="button"
        onClick={() => ref.current?.showModal()}
        className="cursor-pointer underline decoration-muted-foreground/50 underline-offset-4 hover:decoration-foreground"
      >
        {children}
      </button>
      <dialog
        ref={ref}
        onClick={(e) => e.target === ref.current && ref.current.close()}
        className="m-auto w-[min(92vw,32rem)] rounded-md border bg-background p-0 text-foreground shadow-xl backdrop:bg-foreground/20 backdrop:backdrop-blur-[2px]"
      >
        <div className="p-7">
          <div className="flex items-baseline justify-between">
            <h2 className="text-2xl font-medium tracking-[-0.03em]">Peak WAR</h2>
            <button type="button" onClick={() => ref.current?.close()} className="cursor-pointer font-mono text-xs text-muted-foreground hover:text-foreground">
              Close
            </button>
          </div>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            Wins Above Replacement: how many games a season a player wins his team over a replacement-level player. Peak WAR is his
            average over his three best NBA seasons, or all of them if he has played fewer. It measures how good a player became, so
            a career cut short by injury is judged on what it was, and a rookie&apos;s one season counts the same way a veteran&apos;s
            best three do. A player who never played ranks below everyone who did.
          </p>
          <p className="mt-5 rounded-md border bg-muted/50 px-4 py-3 font-mono text-[13px]">
            WAR = 0.000514 × (rating + 2.75) × minutes
          </p>
          <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-5 gap-y-2 text-sm">
            {INPUTS.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="pt-0.5 font-mono text-xs text-muted-foreground">{k}</dt>
                <dd className="leading-6">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </dialog>
    </>
  );
}
