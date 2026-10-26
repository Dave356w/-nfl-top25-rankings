"""Paired portfolio comparison after chronological marginal calibration."""
import csv
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'backtest_output/exposure_2026';OUT=ROOT/'backtest_output/calibration_2026'
settings=['0.75/0.25','1/0.25','0.75/1','1/1'];seeds=list(range(356,366))
games=sorted(p.name.removesuffix('.input.json') for p in OUT.glob('*.input.json'));assert len(games)==17
scores={};raw=[];runs={}
for model,directory in [('baseline',BASE),('calibrated',OUT)]:
    for game in games:
        for seed in seeds:
            r=json.loads((directory/f'{game}.seed{seed}.result.json').read_text());runs[model,game,seed]=r
            assert len(r['rows'])==4 and all(x['filled']==20 for x in r['rows'])
            for row in r['rows']:
                raw.append(dict(model=model,game_id=game,matchup=r['audit']['matchup'],seed=seed,
                    setting=row['setting'],filled=row['filled'],actual_best=row['actual_best'],actual_mean=row['actual_mean'],
                    simulated_best=row['evaluation']['evaluation_best_mean']))
    for setting in settings:
        scores[model,setting]=np.array([[next(x['actual_best'] for x in runs[model,g,s]['rows'] if x['setting']==setting) for s in seeds] for g in games])
with (OUT/'portfolio_comparison_by_game_seed.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
ix=np.random.default_rng(356).integers(0,17,size=(30000,17))
summary={}
for setting in settings:
    b=scores['baseline',setting];c=scores['calibrated',setting];d=c-b;gd=d.mean(axis=1);sd=d.mean(axis=0)
    summary[setting]=dict(baseline_actual_best=float(b.mean()),calibrated_actual_best=float(c.mean()),
        calibrated_minus_baseline=float(d.mean()),game_bootstrap_95=list(map(float,np.quantile(gd[ix].mean(axis=1),[.025,.975]))),
        game_wins=int(sum(gd>1e-8)),game_ties=int(sum(abs(gd)<=1e-8)),game_losses=int(sum(gd< -1e-8)),
        positive_seed_means=int(sum(sd>1e-8)),seed_mean_range=[float(sd.min()),float(sd.max())])
exposure={}
for setting in settings[1:]:
    d=scores['calibrated',setting]-scores['calibrated',settings[0]];gd=d.mean(axis=1)
    exposure[setting]=dict(lift=float(d.mean()),game_bootstrap_95=list(map(float,np.quantile(gd[ix].mean(axis=1),[.025,.975]))),
        positive_seed_means=int(sum(d.mean(axis=0)>1e-8)),game_wins=int(sum(gd>1e-8)),game_ties=int(sum(abs(gd)<=1e-8)),game_losses=int(sum(gd< -1e-8)))
result=dict(games=17,seeds=seeds,entries=20,baseline_vs_calibrated=summary,calibrated_exposure_vs_defaults=exposure,
    interpretation='Expanding historical calibration; same frozen means, pool and prices; exploratory, no untouched future outcomes.')
(OUT/'portfolio_comparison_summary.json').write_text(json.dumps(result,indent=2))
forecast=json.loads((OUT/'simulation_evaluation.json').read_text());audit=json.loads((OUT/'audit_summary.json').read_text());v=forecast['marginals']['ALL']
correlations=list(csv.DictReader((OUT/'correlation_validation.csv').open()));flagged=[r for r in correlations if r['historical_target_inside_interval']=='False']
lines=['# Simulation calibration and portfolio replay — 2026','',
    '## Outcome','',
    'Participation and omission audits, chronological scoring-distribution calibration, correlation diagnostics and paired portfolio comparisons are complete. The calibrated numerical model remains an opt-in research candidate; production projections and exposure defaults are not promoted by this report.','',
    '## Participation and omitted-player audit','',
    f"- {audit['offensive_player_games']} offensive player-games; {audit['matched_snaps']} uniquely matched snap records.",
    f"- {audit['zero_with_offensive_snaps']} zero scores occurred despite offensive snaps; {audit['zero_without_offensive_snaps_but_other_snaps']} had only other snaps. Participation remains unknown for {audit['zero_participation_unknown']} zero-score rows; absence of a snap row is not called inactivity.",
    '- The legacy zero-rate fit required a carry, target or pass attempt. That population omits active players with no recorded opportunity. A positive projection is not a guarantee of participation or scoring opportunity.',
    '- The Sleeper parser drops unsupported primary positions even when fantasy eligibility includes WR or RB. Current source records demonstrate this for Travis Hunter (DB, WR eligible) and Hunter Luepke (FB, RB eligible). The parser fix retains one unambiguous supported fantasy position.',
    '- Ten nonzero settled outcomes occurred outside the current-era frozen pools. Archived removal reasons are in omitted_player_audit.csv. Other reasons include absent projections and later role/pool exclusions whose exact rationale was not archived.',
    '- Historical Sleeper player/projection inputs were not retained. No omitted player is inserted retrospectively from current feeds or postgame scores. New archives retain raw Sleeper inputs and explicit role exclusions.','',
    '## Calibration design','',
    '- Five expanding fits, one before each evaluated weekly decision block. Only outcomes from games with kickoff at least 24 hours before the block cutoff enter training. No evaluated game enters its own fit.',
    '- The six earlier legacy-model games supply zero/negative frequencies as warm-up data. Positive-tail residuals use only earlier Sleeper-projection games. Each fit lists its training games and cutoff.',
    '- Fixed shrinkage strength: 20 observations per role/position cell, chosen before the calibrated portfolio replay. It was not optimized to portfolio scores.',
    '- Each marginal has a zero atom, a position-pooled negative atom, and a positive lognormal tail. Archived means are preserved exactly. Total variance is recalculated from the fitted mixture instead of retaining an incompatible old CV.',
    '- Player pools, salaries, caps, exposure settings, candidate settings and entry counts are unchanged. The positive tail remains continuous, negative magnitude is a coarse pooled atom, and simulated team totals remain unconstrained.','',
    '## Forecast distribution validation','',
    '| Metric | Baseline | Chronologically calibrated |','|---|---:|---:|',
    f"| Expected zero scores (111 observed) | {v['baseline']['expected_zeros']:.2f} | {v['calibrated']['expected_zeros']:.2f} |",
    f"| Expected negative scores (5 observed) | {v['baseline']['expected_negatives']:.2f} | {v['calibrated']['expected_negatives']:.2f} |",
    f"| Zero-probability Brier error, lower better | {v['baseline']['zero_brier']:.4f} | {v['calibrated']['zero_brier']:.4f} |",
    f"| Distribution CRPS, lower better | {v['baseline']['crps']:.4f} | {v['calibrated']['crps']:.4f} |",
    f"| Inclusive P90 coverage | {v['baseline']['p90_coverage']:.1%} | {v['calibrated']['p90_coverage']:.1%} |",'']
for metric,change in forecast['paired_forecast_score_changes'].items():
    lo,hi=change['game_bootstrap_95'];lines.append(f"- {metric}: calibrated minus baseline {change['calibrated_minus_baseline']:+.5f}; paired game-bootstrap 95% [{lo:+.5f}, {hi:+.5f}].")
lines+=['','Means and mean-forecast MAE are unchanged. Better zero calibration is clear in this sample; overall CRPS improvement is small and uncertain.','',
    '## Correlation validation','',
    f"- Historical targets fall inside exploratory game-block intervals for {forecast['correlation_groups_target_inside_interval']} of {forecast['correlation_groups']} pooled relationship groups. This is compatibility with noisy data, not proof of correctness.",
    '- Opposing RB–DEF is flagged: observed standardized residual correlation is about -0.367 versus an average historical target of -0.129 (exploratory 95% interval about -0.513 to -0.153). Multiple relationships were inspected and the sample has only 17 games; this is a follow-up target, not an automatic new coefficient.',
    '- Historical score-correlation targets are retained. New marginals get a new Gaussian latent matrix through numerical inversion and PSD repair; retaining the old latent matrix would silently change score correlations.',
    '- Independent 100,000-scenario checks per game are saved in correlation_simulation_check.json. These check numerical implementation, not empirical football accuracy.','',
    '## Portfolio replay','',
    '17 games × 10 paired seeds × four exposure settings × 20 entries for each model. Standard accuracy: 10,000 scenarios and 10,000 candidate setting. Each cell is the realized best entry in the 20-entry set, averaged over games and seeds.','',
    '| Player / Superstar | Baseline actual best FP | Calibrated actual best FP | Change | Game-bootstrap 95% | Positive seed averages | Games W / T / L |',
    '|---|---:|---:|---:|---|---:|---|']
for setting,s in summary.items():
    lo,hi=s['game_bootstrap_95'];lines.append(f"| {setting} | {s['baseline_actual_best']:.2f} | {s['calibrated_actual_best']:.2f} | {s['calibrated_minus_baseline']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {s['positive_seed_means']}/10 | {s['game_wins']} / {s['game_ties']} / {s['game_losses']} |")
lines+=['','Within the calibrated model, exposure changes versus 0.75 / 0.25:','',
    '| Setting | Best-score change | Game-bootstrap 95% | Positive seed averages |','|---|---:|---|---:|']
for setting,s in exposure.items():
    lo,hi=s['game_bootstrap_95'];lines.append(f"| {setting} | {s['lift']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {s['positive_seed_means']}/10 |")
lines+=['','## Decision and limits','',
    '- Keep the demonstrated parser and archive fixes reviewable as ordinary code changes.',
    '- Keep the calibrated numerical model opt-in. Zero calibration improves, but broader distribution scores, portfolio outcomes, correlation uncertainty and missing-pool provenance do not support calling it fully validated.',
    '- The chronological fitting guards against using each game in its own calibration. This research design was developed after viewing this season, so it is not an untouched prospective test.',
    '- Seventeen games are the outcome sample; seeds repeat algorithm draws. Intervals resample games after averaging seeds and are exploratory across settings.',
    '- The final swap search has a 1.5-second wall-clock budget. Saved selected entries are authoritative; exact entry membership can vary across machines.',
    '- Complete opponent fields and payout curves remain absent. Best-score coverage is not tournament ROI.','',
    '## Reproduce','',
    'Run the original replay preparation first, or restore the supplied exposure_2026 evidence. Then:','',
    '```bash','python tools/prepare_calibrated_replay.py','python tools/evaluate_simulation_calibration.py','python tools/run_calibrated_replay.py','python tools/summarize_calibrated_replay.py','```','',
    'The participation audit additionally uses cached nflreadpy 2026 snap counts and player IDs. Raw historical Sleeper inputs do not exist and cannot be reconstructed from current feeds.','']
(OUT/'2026_simulation_calibration_report.md').write_text('\n'.join(lines))
print(json.dumps(result,indent=2))
