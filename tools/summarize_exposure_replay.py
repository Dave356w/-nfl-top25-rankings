"""Summarize paired game-level results; seeds are not independent games."""
import csv
from datetime import datetime
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'backtest_output/exposure_2026'
manifest=json.loads((OUT/'manifest.json').read_text())
files=sorted(OUT.glob('*.result.json'))
runs=[r for p in files if (r:=json.loads(p.read_text()))['seed'] in (356,357,358)]
settings=['0.75/0.25','1/0.25','0.75/1','1/1']
grouped={}
csvrows=[]
for run in runs:
    a=run['audit'];game=a['game_id']
    grouped.setdefault(game,[]).append(run)
    for row in run['rows']:
        csvrows.append(dict(game_id=game,matchup=a['matchup'],kickoff_utc=a['kickoff_utc'],
            cohort=a['cohort'],snapshot_id=a['snapshot_id'],salary_cap=a['salary_cap'],
            seed=run['seed'],setting=row['setting'],entries=row['filled'],actual_best=row['actual_best'],
            actual_mean=row['actual_mean'],hindsight_best_in_model_pool=run['hindsight_best_in_model_pool'],
            gap_to_pool_optimal=run['hindsight_best_in_model_pool']-row['actual_best'],
            best_submitted=a.get('benchmark_score'),
            gap_to_best_submitted=None if a.get('benchmark_score') is None else a['benchmark_score']-row['actual_best'],
            simulated_best=row['evaluation']['evaluation_best_mean'],
            max_player_count=row['max_player_count'],max_superstar_count=row['max_superstar_count']))
assert len(grouped)==23, f'Incomplete games: {len(grouped)}'
assert all(sorted(r['seed'] for r in group)==[356,357,358] for group in grouped.values()), 'Missing seeds'
assert all(row['entries']==20 for row in csvrows), 'Underfilled portfolios'
with (OUT/'game_seed_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)

summary={}
for cohort in ['sleeper','legacy','all']:
    games=[g for g,rr in grouped.items() if cohort=='all' or rr[0]['audit']['cohort']==cohort]
    means={};pergame={}
    for setting in settings:
        rows=[r for r in csvrows if r['game_id'] in games and r['setting']==setting]
        pergame[setting]=np.array([np.mean([r['actual_best'] for r in rows if r['game_id']==g]) for g in games])
        means[setting]=dict(games=len(games),game_seed_runs=len(rows),
            actual_best_mean=float(np.mean(pergame[setting])),
            actual_entry_mean=float(np.mean([r['actual_mean'] for r in rows])),
            gap_to_pool_optimal=float(np.mean([r['gap_to_pool_optimal'] for r in rows])),
            gap_to_best_submitted=float(np.mean([r['gap_to_best_submitted'] for r in rows if r['gap_to_best_submitted'] is not None])),
            simulated_best_mean=float(np.mean([r['simulated_best'] for r in rows])))
    comparisons={}
    for setting in settings[1:]:
        d=pergame[setting]-pergame[settings[0]]
        rng=np.random.default_rng(356)
        boot=np.mean(rng.choice(d,size=(20000,len(d)),replace=True),axis=1)
        comparisons[setting]=dict(mean_best_score_difference=float(np.mean(d)),
            bootstrap_95_interval=list(map(float,np.quantile(boot,[.025,.975]))),
            wins=int(np.sum(d>1e-8)),ties=int(np.sum(abs(d)<=1e-8)),losses=int(np.sum(d< -1e-8)))
    summary[cohort]=dict(settings=means,paired_vs_default=comparisons)

report=dict(method=dict(entries=20,seeds=[356,357,358],detail='standard',simulations=10000,candidates=10000,
    commit=manifest['commit'],worker_sha256=manifest['worker_sha256'],
    snapshots='Latest strictly pregame snapshot per game; all prices, means and uncertainty frozen.',
    actuals='Exact Yahoo settled identities, points, salaries and caps; no zero imputation.',
    objective='Current browser expected-best-score portfolio, default final swap pass enabled.',
    interval='Exploratory paired bootstrap over games, averaging the three seeds within each game.'),
    coverage=dict(pregame_models=manifest['pregame_games'],completed_series=manifest['completed_series'],
        replayed_games=len(grouped),total_portfolios=len(csvrows),total_entries=len(csvrows)*20),
    summary=summary)
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
game_rows=[]
for game, group in sorted(grouped.items(),key=lambda x:datetime.strptime(x[1][0]['audit']['kickoff_utc'],'%a, %d %b %Y %H:%M:%S %z')):
    a=group[0]['audit']
    row=dict(game_id=game,matchup=a['matchup'],kickoff_utc=a['kickoff_utc'],cohort=a['cohort'],salary_cap=a['salary_cap'])
    for setting in settings:
        row[setting]=float(np.mean([r['actual_best'] for run in group for r in run['rows'] if r['setting']==setting]))
    row['difference_1_1_vs_default']=row['1/1']-row['0.75/0.25']
    game_rows.append(row)
with (OUT/'game_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(game_rows[0]));w.writeheader();w.writerows(game_rows)
lines=['# 2026 Showdown exposure replay','',
    'Exploratory retrospective comparison of the current browser optimizer. No live settings were changed.','',
    f"Code pinned to `{manifest['commit']}`; worker SHA-256 `{manifest['worker_sha256']}`.",'',
    '## Coverage and method','',
    f"- {manifest['pregame_games']} completed-game pregame models; {manifest['completed_series']} Yahoo completed single-game series; 23 exact-priced games replayed.",
    '- 17 games use Sleeper projections (September 27–October 25); 6 use the older projection model (September 13–24).',
    '- 20 entries per portfolio; 3 paired seeds (356, 357, 358); Standard accuracy (10,000 scenarios, 10,000 candidates).',
    '- 276 portfolios and 5,520 graded entries. Every portfolio filled all 20 entries.',
    '- Each game/seed shares its scenarios and screened candidate pool across all four exposure settings.',
    '- Latest strictly pregame snapshot per game. Every modeled player matched Yahoo settled points and the exact contest salary; caps matched too.',
    '- Actual points are used only for grading and the hindsight diagnostic, never for selecting entries.',
    '- Current optimizer replay on old forecasts; these are not portfolios actually published before those games.',
    '- The September 9 contest predates usable archived Showdown models. September 10 was excluded because Yahoo lacks an archived player (Tutu Atwell). Another 81 archived games have no completed single-game contest/cap in the endpoint.',
    '- Game-seed runs are not independent games. Intervals resample games after averaging their three seeds.','']
for cohort,title in [('sleeper','Current Sleeper projection era'),('legacy','Older projection era'),('all','All replayed games')]:
    lines += ['## '+title,'','| Player / Superstar limit | Games | Avg actual best entry | Avg actual entry | Gap to pool optimum | Modeled best score |','|---|---:|---:|---:|---:|---:|']
    for setting in settings:
        v=summary[cohort]['settings'][setting]
        lines.append(f"| {setting} | {v['games']} | {v['actual_best_mean']:.2f} | {v['actual_entry_mean']:.2f} | {v['gap_to_pool_optimal']:.2f} | {v['simulated_best_mean']:.2f} |")
    lines += ['','Compared with 0.75 / 0.25, using paired game averages:','',
        '| Setting | Best-score change | Exploratory 95% interval | Wins / ties / losses |','|---|---:|---|---|']
    for setting in settings[1:]:
        d=summary[cohort]['paired_vs_default'][setting];lo,hi=d['bootstrap_95_interval']
        lines.append(f"| {setting} | {d['mean_best_score_difference']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {d['wins']} / {d['ties']} / {d['losses']} |")
    lines += ['']
lines += ['## Per-game actual best scores (three-seed averages)','','| Date | Matchup | Projection era | Verified cap | Default 0.75 / 0.25 | Relaxed 1 / 1 | Difference |','|---|---|---|---:|---:|---:|---:|']
for row in game_rows:
    date=datetime.strptime(row['kickoff_utc'],'%a, %d %b %Y %H:%M:%S %z').strftime('%Y-%m-%d')
    lines.append(f"| {date} | {row['matchup']} | {row['cohort']} | ${row['salary_cap']:.0f} | {row['0.75/0.25']:.2f} | {row['1/1']:.2f} | {row['difference_1_1_vs_default']:+.2f} |")
lines += ['','## Interpretation and limits','',
    '- Higher actual-best score means the portfolio contained a stronger realized entry. It does not measure total contest return.',
    '- The pool optimum is the highest hindsight score among legal rosters of the saved model pool, including every Superstar choice. It is a diagnostic, not a selectable pregame strategy.',
    '- Relaxing caps expands the feasible set, but the greedy selector, candidate screening and final swap heuristic can still produce a worse realized or simulated result.',
    '- The current final swap pass has a 1.5-second wall-clock budget. Results can vary with hardware/load even with the same seed; exact selected lineups are retained.',
    '- The sample is small, games were already known when this comparison was designed, and there is no untouched validation set. Treat findings as exploratory.',
    '- No complete opponent field or payout curves were available. This report makes no ROI, cash-rate or profit claim.',
    '- A full-production accuracy comparison (20,000 scenarios / 25,000 candidates) has not been run.','',
    '## Included evidence','',
    '- `summary.json`: aggregate results and method.',
    '- `game_seed_results.csv`: one row per game, seed and exposure setting.',
    '- `game_summary.csv`: three-seed average per game and exposure setting, with the verified salary cap.',
    '- `manifest.json`: all 105 pregame games and exclusion reasons.',
    '- `*.input.json`: frozen models, exact identity keys, salaries, caps and settled actuals.',
    '- `*.result.json`: all selected entries and their scores.',
    '- `tools/`: replay and aggregation source. Use the pinned repository commit for the worker and original archive.','',
    'Run from the pinned repository root after placing the supplied evidence in `backtest_output/exposure_2026`:','',
    '```bash','python tools/run_exposure_replay.py','python tools/summarize_exposure_replay.py','```','']
(OUT/'2026_showdown_exposure_report.md').write_text('\n'.join(lines))
print(json.dumps(report,indent=2))
