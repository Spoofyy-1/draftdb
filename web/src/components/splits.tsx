import type { Split } from "@/lib/db";

const range = (ys: number[]) => (ys.length === 1 ? `${ys[0]}` : `${ys[0]}–${ys[ys.length - 1]}`);

/** One strip: the context classes, the holdout classes, and how the walk-forward works. */
export function Splits({ splits }: { splits: Split[] }) {
  const holdout = splits.filter((s) => s.role === "holdout").map((s) => s.year);
  const context = splits.filter((s) => s.role === "context" && s.year < holdout[0]).map((s) => s.year);
  if (!holdout.length || !context.length) return null;

  return (
    <div>
      <ol className="flex flex-wrap gap-1">
        {splits.map((s) => (
          <li
            key={s.year}
            className={`rounded px-2 py-1 font-mono text-xs tabular-nums ${
              context.includes(s.year) ? "bg-foreground text-background" : holdout.includes(s.year) ? "ring-1 ring-foreground font-medium" : "text-muted-foreground/40"
            }`}
          >
            {s.year}
          </li>
        ))}
      </ol>
      <p className="mt-4 max-w-2xl text-[15px] leading-7 text-muted-foreground">
        <span className="text-foreground">Context: {range(context)}.</span> <span className="text-foreground">Holdout: {range(holdout)}.</span> Every holdout draft is scored
        walk-forward: the model sees only the drafts before it, labelled with only the NBA seasons those players had played by that
        draft night. So {holdout[0]} learns from {range(context)}; {holdout[1] ?? holdout[0] + 1} also learns from {holdout[0]}&apos;s players, and so on.
        Nothing from a draft&apos;s own year or later is ever visible.
      </p>
    </div>
  );
}
