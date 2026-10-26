"""Auditable pregame-forecast to final Yahoo player/game outcome join."""
from collections import Counter
from datetime import datetime
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.showdown_archive_backtest import completed_player_key, settled_actuals
OUT=ROOT/'backtest_output/exposure_2026'
series={r['series_id']:r for r in json.loads((ROOT/'backtest_cache/completed_2026.json').read_text())}
rows=[];game_rows=[];extras=[];seen=set()
for path in sorted(OUT.glob('*.input.json')):
    job=json.loads(path.read_text());a=job['audit'];game=a['game_id']
    s=series[a['series_id']]
    assert s['game_code']==game and float(s['salary_cap'])==job['salary_cap']
    assert datetime.fromisoformat(a['captured_utc'])<datetime.strptime(a['kickoff_utc'],'%a, %d %b %Y %H:%M:%S %z')
    raw=json.loads(gzip.decompress((ROOT/f"backtest_cache/yahoo/players_{a['series_id']}.json.gz").read_bytes()))
    points,salaries=settled_actuals(raw)
    raw_rows={completed_player_key(r):r for r in raw['players']['result']}
    assert len([k for k in raw_rows if k is not None])==len({completed_player_key(r) for r in raw['players']['result'] if completed_player_key(r) is not None})
    for index,(key,player) in enumerate(zip(job['player_keys'],job['payload']['players'])):
        identity=(game,key)
        assert identity not in seen;seen.add(identity)
        settled=raw_rows[key];g=settled['game']
        assert g['code']==game and g['finished'] is True
        assert g['startTime']==s['kickoff_ms']
        assert settled['playerGameCode']==settled['code']+'$'+game
        assert points[key]==job['actuals'][index]
        assert salaries[key]==float(player['salary'])
        expected=float(player['fp']);actual=float(points[key])
        rows.append(dict(game_id=game,series_id=a['series_id'],matchup=a['matchup'],kickoff_utc=a['kickoff_utc'],
            snapshot_id=a['snapshot_id'],captured_utc=a['captured_utc'],cohort=a['cohort'],
            player_key=key,player=player['name'],pregame_team=player['team'],position=player['pos'],
            depth=player.get('depth'),role=player.get('role'),salary=float(player['salary']),salary_cap=job['salary_cap'],
            projected_fp=expected,actual_yahoo_fp=actual,error_actual_minus_projection=actual-expected,
            absolute_error=abs(actual-expected),projected_cv=player.get('cv'),projected_zero_rate=player.get('zero'),
            actual_zero=int(actual==0),actual_negative=int(actual<0),
            yahoo_game_finished=True,identity_match='exact game ID + Yahoo player key',projection_source=player.get('source')))
    modeled=set(job['player_keys'])
    for key,actual in points.items():
        if key in modeled:continue
        settled=raw_rows[key]
        extras.append(dict(game_id=game,series_id=a['series_id'],player_key=key,
            yahoo_display_name=' '.join(filter(None,[settled.get('firstName'),settled.get('lastName')])),
            actual_yahoo_fp=actual,settled_salary=salaries[key],reason='Not present in frozen modeled player pool'))
    game_rows.append(dict(game_id=game,matchup=a['matchup'],cohort=a['cohort'],series_id=a['series_id'],
        modeled_players=len(modeled),exact_matches=len(modeled),unmatched=0,salary_mismatches=0,
        wrong_game=0,not_final=0,settled_players_outside_model=sum(e['game_id']==game for e in extras)))
for name,data in [('player_game_outcomes.csv',rows),('player_game_match_audit.csv',game_rows),('settled_players_outside_model.csv',extras)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary={}
for cohort in ['sleeper','legacy','all']:
    selected=[r for r in rows if cohort=='all' or r['cohort']==cohort]
    summary[cohort]={}
    for pos in ['ALL','QB','RB','WR','TE','DEF']:
        data=[r for r in selected if pos=='ALL' or r['position']==pos]
        error=np.array([r['error_actual_minus_projection'] for r in data])
        summary[cohort][pos]=dict(player_game_records=len(data),mae=float(np.mean(abs(error))),
            rmse=float(np.sqrt(np.mean(error**2))),bias_actual_minus_projection=float(np.mean(error)),
            observed_zero_rate=float(np.mean([r['actual_zero'] for r in data])),
            observed_negative_rate=float(np.mean([r['actual_negative'] for r in data])))
audit=dict(games=len(game_rows),matched_player_games=len(rows),unmatched=0,salary_mismatches=0,
    wrong_game=0,not_final=0,duplicate_player_game_keys=0,settled_players_outside_model=len(extras),
    current_era_games=sum(r['cohort']=='sleeper' for r in game_rows),
    current_era_player_games=sum(r['cohort']=='sleeper' for r in rows),
    scoring='Yahoo settled DFS points; Superstar multiplier applied only when grading lineups.',
    join='Completed series game code + archived stable player key; per-player game code, exact kickoff, final flag and salary verified.',
    mean_error_definition='Actual minus projected; positive means underprediction.',
    missing_actuals='No zero imputation. Historical exclusions remain in manifest.json.',
    statistical_unit='Player-game rows are correlated within each game; calibration metrics are descriptive.',
    cross_provider_validation='Not performed here; Yahoo final DFS points are the scoring authority.',projection_diagnostics=summary)
current_games={r['game_id'] for r in game_rows if r['cohort']=='sleeper'}
audit['outside_pool_nonzero_records']=sum(r['actual_yahoo_fp']!=0 for r in extras)
audit['current_era_outside_pool_nonzero_records']=sum(r['actual_yahoo_fp']!=0 and r['game_id'] in current_games for r in extras)
(OUT/'player_game_audit_summary.json').write_text(json.dumps(audit,indent=2))
text=['# Actual player outcomes matched to games','',
    f"Matched **{len(rows)} player-game records across {len(game_rows)} games**, including {audit['current_era_player_games']} records across 17 games using Sleeper projections.",'',
    '**No unmatched modeled players, salary mismatches, wrong-game records, unfinished games or duplicate player-game keys.**','',
    'Each join verifies the completed contest game ID, stable Yahoo player key, per-player game ID, exact kickoff, final flag, salary and settled fantasy points. The forecast snapshot must be strictly before kickoff.','',
    '- `player_game_outcomes.csv`: archived projection, position, role, salary, uncertainty parameters, final Yahoo FP and errors for every modeled player/game.',
    '- `player_game_match_audit.csv`: matching checks summarized by game.',
    '- `settled_players_outside_model.csv`: additional Yahoo settled players outside the frozen modeled pool; excluded from optimizer replay.',
    '- `player_game_audit_summary.json`: counts and descriptive projection diagnostics by position and model era.','',
    '## Current Sleeper era: descriptive forecast error','',
    '| Position | Player-games | MAE | RMSE | Mean actual minus projection |','|---|---:|---:|---:|---:|']
for pos,v in summary['sleeper'].items():
    text.append(f"| {pos} | {v['player_game_records']} | {v['mae']:.2f} | {v['rmse']:.2f} | {v['bias_actual_minus_projection']:+.2f} |")
text+=['','Positive mean error indicates underprediction. These rows share football games and are not independent trials.','',
    '## Player-pool coverage follow-up','',
    f"Yahoo has {len(extras)} settled player-game records outside the archived modeled pools. Of those, {audit['outside_pool_nonzero_records']} scored nonzero points; {audit['current_era_outside_pool_nonzero_records']} are in the Sleeper era. They are listed separately and were available to none of the compared strategies.",
    'Examples include Travis Hunter (13.9 FP, HOU vs JAC) and Athan Kaliakmanis (12.04 FP, IND vs WAS). Their absence needs a separate availability/player-pool audit; this join does not establish that they should have been selectable before kickoff.','',
    'The replay and stability report grade selected portfolios from these actual outcomes. This confirms the retrospective scoring inputs; it does not create new validation games or establish tournament profitability.','',
    'Yahoo settled DFS points are used directly. No independent NFLverse cross-provider scoring audit was performed in this join.','',
    'Run `python tools/audit_player_game_outcomes.py` from the pinned repository to regenerate.','']
(OUT/'2026_player_game_outcome_audit.md').write_text('\n'.join(text))
print(json.dumps({k:v for k,v in audit.items() if k!='projection_diagnostics'},indent=2))
