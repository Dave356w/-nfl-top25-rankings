"""Prepare exact-priced 2026 replay inputs from latest strictly pregame snapshots."""
import concurrent.futures
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from pipeline.showdown_backtest import YahooDfsClient
from pipeline.showdown_archive_backtest import model_player_keys, select_completed_series

OUT = ROOT / 'backtest_output/exposure_2026'
OUT.mkdir(parents=True, exist_ok=True)
series_path = ROOT / 'backtest_cache/completed_2026.json'
if not series_path.exists():
    client=YahooDfsClient(ROOT/'backtest_cache/yahoo',timeout=20,pause=0)
    discovered=client.discover(date(2026,9,1),date(2026,10,27))
    series_path.parent.mkdir(parents=True,exist_ok=True)
    series_path.write_text(json.dumps(discovered,indent=2))
series = json.loads(series_path.read_text())
by_game = {}
for row in series:
    by_game.setdefault(row['game_code'], []).append(row)
latest = {}
now = pd.Timestamp('2026-10-26T23:00:00Z')  # fixed replay horizon, before Week 7 MNF
for path in (ROOT / 'site/data/projection_archive/snapshots').glob('*.gz'):
    snapshot = json.loads(gzip.decompress(path.read_bytes()))
    captured = pd.Timestamp(snapshot['captured_utc'])
    for model in snapshot.get('showdown_models', []):
        kickoff = pd.Timestamp(model['kickoff_utc'])
        if kickoff.year != 2026 or kickoff >= now or captured >= kickoff:
            continue
        game = model['game_id']
        if game not in latest or captured > latest[game][0]:
            latest[game] = (captured, snapshot, model, path.name)

def prepare(item):
    game, (captured, snapshot, model, archive_file) = item
    audit = dict(game_id=game, matchup=model['matchup'], kickoff_utc=model['kickoff_utc'],
                 captured_utc=str(captured), snapshot_id=snapshot['snapshot_id'], archive_file=archive_file)
    try:
        if game not in by_game:
            raise ValueError('No completed Yahoo single-game series; no verified contest cap')
        keys = model_player_keys(snapshot, model)
        client = YahooDfsClient(ROOT / 'backtest_cache/yahoo', timeout=15, pause=0)
        settled, raw, actuals = select_completed_series(model, by_game[game], client, keys)
        cap = float(settled['salary_cap'])
        if model.get('salary_cap') is not None and abs(float(model['salary_cap']) - cap) > 1e-9:
            raise ValueError('Archived and completed salary caps disagree')
        benchmark = None
        try:
            benchmark = (client.benchmark(settled['series_id']).get('allUserBestLineup', {}).get('result') or {}).get('score')
        except Exception as exc:
            audit['benchmark_error'] = str(exc)
        sources = sorted({p.get('source', '') for p in model['players']})
        audit.update(series_id=settled['series_id'], salary_cap=cap,
                     salary_cap_source='pregame_archive' if model.get('salary_cap') is not None else 'completed_yahoo_series',
                     player_count=len(keys), sources=sources, actual_source='yahoo_settled',
                     cohort='sleeper' if all('Sleeper' in p.get('source','') for p in model['players']) else 'legacy',
                     benchmark_score=benchmark, status='ready')
        job=dict(audit=audit, payload=model, salary_cap=cap, player_keys=keys,
                 actuals=[float(actuals[k]) for k in keys])
        (OUT / (game + '.input.json')).write_text(json.dumps(job))
    except Exception as exc:
        audit.update(status='skipped', reason=str(exc))
    return audit

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    audits=list(pool.map(prepare, sorted(latest.items())))
audits.sort(key=lambda x: pd.Timestamp(x['kickoff_utc']))
manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              worker_sha256=hashlib.sha256((ROOT/'site/showdown-worker.js').read_bytes()).hexdigest(),
              prepared_utc=str(now), pregame_games=len(latest), completed_series=len(series), games=audits)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
for a in audits:
    if a['status']=='ready' or a['game_id'] in by_game:
        print(a,flush=True)
print('Ready',sum(a['status']=='ready' for a in audits),'of',len(audits),flush=True)
