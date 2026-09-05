"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { TargetKind } from "@/lib/db";

const DEFINITION: Record<TargetKind, { title: string; text: string }> = {
  war3: {
    title: "3-year WAR",
    text:
      "Wins Above Replacement: how many games a season a player wins his team over a replacement-level player. 3-year WAR adds up " +
      "his WAR over his first three NBA seasons, or all of them if he has played fewer, so it measures what a pick produced early in " +
      "his career, not how good he eventually became. Recent classes are judged on the seasons they have played so far. A player who " +
      "never played ranks below everyone who did.",
  },
  peak: {
    title: "Peak WAR",
    text:
      "Wins Above Replacement: how many games a season a player wins his team over a replacement-level player. Peak WAR is his " +
      "average over his three best NBA seasons, or all of them if he has played fewer. It measures how good a player became, so a " +
      "career cut short by injury is judged on what it was, and a rookie's one season counts the same way a veteran's best three do. " +
      "A player who never played ranks below everyone who did.",
  },
};

const INPUTS: [string, string][] = [
  ["rating", "How many points per 100 possessions the player is worth compared with an average player (FiveThirtyEight RAPTOR, or BPM on the same scale after 2022)."],
  ["2.75", "Replacement level. A bench-level player is about 2.75 points per 100 worse than average and earns zero WAR."],
  ["minutes", "Regular-season minutes played that season."],
  ["0.000514", "Converts points and minutes into wins."],
];

/** Inline link that opens a modal explaining the WAR score both draft orders are judged against. */
export function WarModal({ target = "war3", children }: { target?: TargetKind; children: ReactNode }) {
  const def = DEFINITION[target] ?? DEFINITION.war3;
  const ref = useRef<HTMLDialogElement>(null);
  // The trigger sits inside a <p>, which may only contain phrasing content, so the dialog is portalled to body.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const dialog = (
    <dialog
      ref={ref}
      onClick={(e) => e.target === ref.current && ref.current.close()}
      className="m-auto w-[min(92vw,32rem)] rounded-md border bg-background p-0 text-foreground shadow-xl backdrop:bg-foreground/20 backdrop:backdrop-blur-[2px]"
    >
      <div className="p-7">
        <div className="flex items-baseline justify-between">
          <h2 className="text-2xl font-medium tracking-[-0.03em]">{def.title}</h2>
          <button type="button" onClick={() => ref.current?.close()} className="cursor-pointer font-mono text-xs text-muted-foreground hover:text-foreground">
            Close
          </button>
        </div>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">{def.text}</p>
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
  );

  return (
    <>
      <button
        type="button"
        onClick={() => ref.current?.showModal()}
        className="cursor-pointer underline decoration-muted-foreground/50 underline-offset-4 hover:decoration-foreground"
      >
        {children}
      </button>
      {mounted && createPortal(dialog, document.body)}
    </>
  );
}
