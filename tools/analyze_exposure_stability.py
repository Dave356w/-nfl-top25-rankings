"""Game- and seed-level stability diagnostics for the frozen exposure replay."""
import csv
from datetime import datetime
import itertools
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'backtest_output/exposure_2026'
SEEDS=list(range(356,366))
SETTINGS=['0.75/0.25','1/0.25','0.75/1','1/1']
EPS=1e-8
inputs={j['audit']['game_id']:j for p in OUT.glob('*.input.json')
        if (j:=json.loads(p.read_text()))['audit']['cohort']=='sleeper'}
games=sorted(inputs,key=lambda g:datetime.strptime(inputs[g]['audit']['kickoff_utc'],'%a, %d %b %Y %H:%M:%S %z'))
assert len(games)==17
runs={}
for game in games:
    for seed in SEEDS:
        r=json.loads((OUT/f'{game}.seed{seed}.result.json').read_text())
        assert r['seed']==seed and r['audit']['game_id']==game
        assert all(x['filled']==20 and x['requested']==20 for x in r['rows'])
        runs[game,seed]=r
scores={setting:np.array([[next(x['actual_best'] for x in runs[g,s]['rows'] if x['setting']==setting)
                           for s in SEEDS] for g in games]) for setting in SETTINGS}
# The original published three-seed results must remain reproducible exactly.
original=json.loads((OUT/'summary.json').read_text())['summary']['sleeper']['settings']
for setting in SETTINGS:
    assert abs(scores[setting][:,:3].mean()-original[setting]['actual_best_mean'])<1e-9
weeks=[int(re.search(r'week (\d+)',inputs[g]['payload']['players'][0]['source']).group(1)) for g in games]
unique_weeks=sorted(set(weeks))
game_indices=np.random.default_rng(356).integers(0,len(games),size=(30000,len(games)))
week_indices=np.random.default_rng(357).integers(0,len(unique_weeks),size=(30000,len(unique_weeks)))

def wtl(a):
    return dict(wins=int(np.sum(a>EPS)),ties=int(np.sum(abs(a)<=EPS)),losses=int(np.sum(a< -EPS)))

def comparison(candidate,baseline):
    diff=scores[candidate]-scores[baseline]
    per_game=diff.mean(axis=1);per_seed=diff.mean(axis=0)
    boot=per_game[game_indices].mean(axis=1)
    sums=np.array([per_game[np.array(weeks)==w].sum() for w in unique_weeks])
    counts=np.array([sum(x==w for x in weeks) for w in unique_weeks])
    week_boot=sums[week_indices].sum(axis=1)/counts[week_indices].sum(axis=1)
    ordered=np.sort(per_game)
    loo=(per_game.sum()-per_game)/(len(games)-1)
    return dict(candidate=candidate,baseline=baseline,mean_lift=float(diff.mean()),
        median_game_lift=float(np.median(per_game)),game_wtl=wtl(per_game),
        game_seed_wtl=wtl(diff),positive_seed_means=int(sum(per_seed>EPS)),
        seed_mean_range=[float(per_seed.min()),float(per_seed.max())],
        seed_means={str(s):float(x) for s,x in zip(SEEDS,per_seed)},
        game_bootstrap_95=list(map(float,np.quantile(boot,[.025,.975]))),
        week_bootstrap_95=list(map(float,np.quantile(week_boot,[.025,.975]))),
        leave_one_game_out_range=[float(loo.min()),float(loo.max())],
        lift_after_removing_two_best_games=float(ordered[:-2].mean()),
        lift_after_removing_three_best_games=float(ordered[:-3].mean()),
        worst_game_lift=float(per_game.min()),best_game_lift=float(per_game.max()),
        weekly_mean_lift={str(w):float(per_game[np.array(weeks)==w].mean()) for w in unique_weeks},
        games_win_at_least_7_seeds=int(sum((diff>EPS).sum(axis=1)>=7)),
        games_lose_at_least_7_seeds=int(sum((diff< -EPS).sum(axis=1)>=7)))

setting_summary={}
for setting in SETTINGS:
    retention=[]
    for g in games:
        sets=[]
        for s in SEEDS:
            row=next(x for x in runs[g,s]['rows'] if x['setting']==setting)
            sets.append({(tuple(sorted(x['ids'])),x['superstar']) for x in row['lineups']})
        retention.extend(len(a&b)/20 for a,b in itertools.combinations(sets,2))
    setting_summary[setting]=dict(mean_best_actual=float(scores[setting].mean()),
        mean_within_game_seed_sd=float(scores[setting].std(axis=1,ddof=1).mean()),
        median_within_game_seed_sd=float(np.median(scores[setting].std(axis=1,ddof=1))),
        mean_lineup_retention_across_seed_pairs=float(np.mean(retention)))
comparisons={s:comparison(s,SETTINGS[0]) for s in SETTINGS[1:]}
direct=comparison('1/1','0.75/1')
method=dict(games=17,seeds=SEEDS,weekly_blocks=unique_weeks,entries=20,
    simulations=10000,candidate_setting=10000,portfolios=680,graded_entries=13600,
    inference_unit='17 games; 170 game/seed runs are repeated algorithm evaluations.',
    score='Best realized Yahoo FP among each 20-entry portfolio.',
    game_interval='30,000 paired game bootstrap samples after averaging all ten seeds within each game.',
    week_interval='30,000 paired week-block bootstrap samples, preserving all games within each sampled week.',
    limitations=['Exploratory retrospective comparison; no untouched validation games.',
                 'Only 17 games across five weeks; bootstrap intervals are descriptive sensitivity checks.',
                 'The final worker swap pass has a 1.5-second wall-clock limit; original selected entries are preserved.',
                 'Lineup membership changes can coexist with stable realized scores.',
                 'Fantasy-point coverage is not tournament ROI.'])
result=dict(method=method,settings=setting_summary,vs_default=comparisons,
    fully_relaxed_vs_superstar_only=direct)
(OUT/'stability_summary.json').write_text(json.dumps(result,indent=2))

seed_rows=[];game_rows=[]
for setting in SETTINGS:
    diff=scores[setting]-scores[SETTINGS[0]]
    for s,seed in enumerate(SEEDS):
        seed_rows.append(dict(setting=setting,seed=seed,games=17,
            mean_best_actual=float(scores[setting][:,s].mean()),mean_lift_vs_default=float(diff[:,s].mean()),**wtl(diff[:,s])))
    for i,g in enumerate(games):
        a=inputs[g]['audit'];d=diff[i,:]
        game_rows.append(dict(game_id=g,matchup=a['matchup'],week=weeks[i],setting=setting,seeds=10,
            mean_best_actual=float(scores[setting][i,:].mean()),mean_lift_vs_default=float(d.mean()),
            seed_sd=float(scores[setting][i,:].std(ddof=1)),**wtl(d)))
for name,rows in [('stability_by_seed.csv',seed_rows),('stability_by_game.csv',game_rows)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

lines=['# 2026 Showdown exposure stability','',
    'Expanded analysis of the 17 games using archived Sleeper projections, September 27–October 25, 2026.','',
    'Ten paired seeds (356–365), 20 entries per portfolio, Standard accuracy. This is 680 portfolios and 13,600 graded entries across 17 games. Original three-seed results are retained in the companion report.','',
    '## Overall and seed stability','',
    '| Player / Superstar | Avg actual best FP | Lift vs default | Positive seed averages | Seed-average lift range | Games W / T / L |',
    '|---|---:|---:|---:|---|---|']
for s in SETTINGS:
    v=setting_summary[s]
    if s==SETTINGS[0]:
        lines.append(f"| {s} | {v['mean_best_actual']:.2f} | — | — | — | — |")
    else:
        c=comparisons[s];lo,hi=c['seed_mean_range'];w=c['game_wtl']
        lines.append(f"| {s} | {v['mean_best_actual']:.2f} | {c['mean_lift']:+.2f} | {c['positive_seed_means']}/10 | {lo:+.2f} to {hi:+.2f} | {w['wins']} / {w['ties']} / {w['losses']} |")
lines+=['','Wins/ties/losses compare each game after averaging its ten seeds. A positive seed average means a setting improved the average over all 17 games for that seed.','',
    '## Concentration and uncertainty','',
    '| Setting | Game bootstrap 95% | Week-block bootstrap 95% | Remove best 2 games | Remove best 3 games | Leave-one-game-out range |',
    '|---|---|---|---:|---:|---|']
for s,c in comparisons.items():
    gl,gh=c['game_bootstrap_95'];wl,wh=c['week_bootstrap_95'];ll,lh=c['leave_one_game_out_range']
    lines.append(f"| {s} | [{gl:+.2f}, {gh:+.2f}] | [{wl:+.2f}, {wh:+.2f}] | {c['lift_after_removing_two_best_games']:+.2f} | {c['lift_after_removing_three_best_games']:+.2f} | {ll:+.2f} to {lh:+.2f} |")
lines+=['','The removal tests exclude the largest improving games for each setting; they are deliberate stress tests, not new unbiased estimates. Bootstrap intervals are exploratory and do not adjust for trying several settings.','',
    '## Fully relaxed versus Superstar-only relaxation','']
d=direct;w=d['game_wtl'];lo,hi=d['game_bootstrap_95']
lines += [f"1 / 1 minus 0.75 / 1: **{d['mean_lift']:+.3f} FP per game**; game bootstrap 95% [{lo:+.2f}, {hi:+.2f}].",
    f"Positive average in {d['positive_seed_means']}/10 seeds. Game wins/ties/losses: {w['wins']} / {w['ties']} / {w['losses']}.",'',
    '## Score variability and lineup membership','',
    '| Setting | Avg within-game SD across seeds | Avg entries shared between two seeds |','|---|---:|---:|']
for s,v in setting_summary.items():
    lines.append(f"| {s} | {v['mean_within_game_seed_sd']:.2f} FP | {100*v['mean_lineup_retention_across_seed_pairs']:.1f}% |")
lines+=['','Shared entries require the same five players and Superstar. This measures membership reproducibility, not lineup quality. A different lineup set can deliver the same realized best score.','',
    '## Per-seed gains versus default','',
    '| Seed | Player only 1 / 0.25 | Superstar only 0.75 / 1 | Both 1 / 1 |','|---|---:|---:|---:|']
for seed in SEEDS:
    lines.append('| '+str(seed)+' | '+' | '.join(f"{comparisons[s]['seed_means'][str(seed)]:+.2f}" for s in SETTINGS[1:])+' |')
lines+=['','## Per-week mean gains versus default','',
    '| NFL week | Games | Player only | Superstar only | Both |','|---|---:|---:|---:|---:|']
for week in unique_weeks:
    lines.append('| '+str(week)+' | '+str(weeks.count(week))+' | '+' | '.join(f"{comparisons[s]['weekly_mean_lift'][str(week)]:+.2f}" for s in SETTINGS[1:])+' |')
lines+=['','## Per-game gains versus default','',
    '| Game | Week | Superstar-only lift | Superstar-only seeds W / T / L | Both lift | Both seeds W / T / L |','|---|---:|---:|---|---:|---|']
for i,g in enumerate(games):
    values=[]
    for s in ['0.75/1','1/1']:
        diff=scores[s][i,:]-scores[SETTINGS[0]][i,:];w=wtl(diff)
        values.extend([f'{diff.mean():+.2f}',f"{w['wins']} / {w['ties']} / {w['losses']}"])
    lines.append('| '+inputs[g]['audit']['matchup']+' | '+str(weeks[i])+' | '+' | '.join(values)+' |')
lines+=['','## Method and limitations','',
    '- Player forecasts, salaries, caps and uncertainty inputs remain frozen to the latest strictly pregame snapshot per game.',
    '- The current worker is pinned to repository commit 566f78628508c4bdfa45abb6db5fd93f64fe48a0. Every setting within each game/seed shares the same scenarios and candidate pool.',
    '- Exact Yahoo settled identities, salaries and points are used. Missing actuals were not filled with zero.',
    '- Game-level inference averages seeds within each game. The week-block sensitivity check resamples the five NFL weeks as blocks.',
    '- Seed means show simulation sensitivity on these fixed games; they do not represent additional football outcomes.',
    '- No setting was chosen using an untouched validation set. Findings remain retrospective; use future games for confirmation.',
    '- The worker final swap pass is time-limited and may vary with hardware/load. The resulting lineups are retained exactly.',
    '- ROI, cash rate and prize splitting cannot be established from these best-score comparisons.','',
    'Reproduce after restoring the included data under backtest_output/exposure_2026 in the pinned repository:','',
    '```bash','python tools/run_exposure_stability.py','python tools/analyze_exposure_stability.py','```','']
(OUT/'2026_showdown_stability_report.md').write_text('\n'.join(lines))
print(json.dumps(result,indent=2))
