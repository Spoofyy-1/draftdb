-- Postgres schema for the redraft results the web app reads (replaces outputs/results.sqlite).
-- Apply once per project: Supabase dashboard -> SQL editor, or `python -m validation.db --schema` with SUPABASE_DB_URL set.
-- Safe to re-run.

create table if not exists public.runs (
  run_id            text primary key,
  created           text,
  tag               text,
  models            text,
  redraft_model     text,
  north_star        text,
  target            text,
  target_desc       text,
  feature_hash      text,
  n_features        integer,
  holdout_evaluated integer,
  gpus              integer
);

create table if not exists public.splits (
  run_id         text not null references public.runs(run_id) on delete cascade,
  year           integer not null,
  role           text,
  labelled       integer,
  causal_context jsonb,
  primary key (run_id, year)
);

create table if not exists public.metrics (
  run_id              text not null references public.runs(run_id) on delete cascade,
  split               text,
  protocol            text,
  model               text,
  year                integer,
  n_context           integer,
  n                   integer,
  seconds             double precision,
  spearman            double precision,
  spearman_nba        double precision,
  war_captured_pct_14 double precision,
  war_captured_pct_30 double precision,
  ndcg_14             double precision,
  ndcg_30             double precision,
  war_ours_14         double precision,
  war_actual_14       double precision,
  war_oracle_14       double precision,
  war_ours_30         double precision,
  war_actual_30       double precision,
  war_oracle_30       double precision
);

create table if not exists public.redraft_picks (
  run_id         text not null references public.runs(run_id) on delete cascade,
  year           integer,
  model          text,
  protocol       text,
  player         text,
  bbref_id       text,
  actual_pick    integer,
  new_pick       integer,
  team           text,
  college        text,
  modelled       integer,
  pred           double precision,
  war            double precision,
  seasons_played integer,
  labelled       integer
);

create index if not exists ix_metrics on public.metrics (run_id, split, protocol, model);
create index if not exists ix_picks on public.redraft_picks (run_id, year);
create index if not exists ix_picks_model on public.redraft_picks (run_id, model, protocol);

-- The site reads with the publishable (anon) key, so every table needs RLS on with a read-only policy.
-- Writes go through the service role key, which bypasses RLS.
alter table public.runs enable row level security;
alter table public.splits enable row level security;
alter table public.metrics enable row level security;
alter table public.redraft_picks enable row level security;

drop policy if exists anon_read on public.runs;
drop policy if exists anon_read on public.splits;
drop policy if exists anon_read on public.metrics;
drop policy if exists anon_read on public.redraft_picks;

create policy anon_read on public.runs          for select to anon, authenticated using (true);
create policy anon_read on public.splits        for select to anon, authenticated using (true);
create policy anon_read on public.metrics       for select to anon, authenticated using (true);
create policy anon_read on public.redraft_picks for select to anon, authenticated using (true);
