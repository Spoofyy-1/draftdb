import type { Split } from "@/lib/db";

const range = (ys: number[]) => (ys.length === 1 ? `${ys[0]}` : `${ys[0]}–${ys[ys.length - 1]}`);

/** One strip: the training classes, the test classes, and how the walk-forward works. */
export function Splits({ splits }: { splits: Split[] }) {
  const test = splits.filter((s) => s.role === "validation" || s.role === "holdout").map((s) => s.year);
  const train = splits.filter((s) => s.role === "context" && s.year < test[0]).map((s) => s.year);
  if (!test.length || !train.length) return null;

  return (
    <div>
      <ol className="flex flex-wrap gap-1">
        {splits.map((s) => (
          <li
            key={s.year}
            className={`rounded px-2 py-1 font-mono text-xs tabular-nums ${
              train.includes(s.year) ? "bg-foreground text-background" : test.includes(s.year) ? "ring-1 ring-foreground font-medium" : "text-muted-foreground/40"
            }`}
          >
            {s.year}
          </li>
        ))}
      </ol>
      <p className="mt-4 max-w-2xl text-[15px] leading-7 text-muted-foreground">
        <span className="text-foreground">Training: {range(train)}.</span> <span className="text-foreground">Test: {range(test)}.</span> Every test draft is scored
        walk-forward: the model sees only the drafts before it, labelled with only the NBA seasons those players had played by that
        draft night. So {test[0]} learns from {range(train)}; {test[1] ?? test[0] + 1} also learns from {test[0]}&apos;s players, and so on.
        Nothing from a draft&apos;s own year or later is ever visible.
      </p>
    </div>
  );
}
