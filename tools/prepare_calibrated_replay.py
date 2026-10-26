"""Audit omissions/participation and build strictly chronological research payloads."""
import csv
import gzip
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline import simulation_calibration as sc
from pipeline import notebook as nb
SOURCE=ROOT/'backtest_output/exposure_2026'
OUT=ROOT/'backtest_output/calibration_2026';OUT.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((SOURCE/'player_game_outcomes.csv').open()))
for r in rows:
    for k in ['projected_fp','actual_yahoo_fp','salary','projected_cv','projected_zero_rate']:r[k]=float(r[k])
    r['depth']=int(r['depth'])
jobs={j['audit']['game_id']:j for p in SOURCE.glob('*.input.json') if (j:=json.loads(p.read_text()))['audit']['cohort']=='sleeper'}
weeks={g:int(re.search(r'week (\d+)',j['payload']['players'][0]['source']).group(1)) for g,j in jobs.items()}
artifacts={}
for week in sorted(set(weeks.values())):
    cutoff=min(pd.Timestamp(jobs[g]['payload']['kickoff_utc']) for g in jobs if weeks[g]==week)
    training=[r for r in rows if pd.Timestamp(r['kickoff_utc'])+pd.Timedelta(hours=24)<cutoff]
    artifact=sc.fit(training,cutoff)
    artifacts[week]=artifact
    (OUT/f'fit_before_week_{week}.json').write_text(json.dumps(artifact,indent=2))
    print('Week',week,'training games',len(artifact['training_games']),'rows',len(training),flush=True)
for game,job in jobs.items():
    artifact=artifacts[weeks[game]]
    payload=sc.calibrated_payload(job['payload'],artifact)
    job['payload']=payload;job['audit']['calibration_family']=artifact['family']
    job['audit']['calibration_cutoff']=artifact['cutoff_utc'];job['audit']['training_games']=artifact['training_games']
    (OUT/f'{game}.input.json').write_text(json.dumps(job))

# Explain all observed pool omissions from pregame audit trails, never postgame projections.
manifest=json.loads((SOURCE/'manifest.json').read_text());outside=list(csv.DictReader((SOURCE/'settled_players_outside_model.csv').open()))
omissions=[]
for game,job in jobs.items():
    a=next(a for a in manifest['games'] if a['game_id']==game)
    snapshot=json.loads(gzip.decompress((ROOT/'site/data/projection_archive/snapshots'/a['archive_file']).read_bytes()))
    for r in outside:
        if r['game_id']!=game or float(r['actual_yahoo_fp'])==0:continue
        name=r['yahoo_display_name']
        candidates=[x for x in snapshot.get('availability_removed',[]) if x['Player']==name]
        if len(candidates)==1:
            reason=candidates[0]['Reason'];injury=candidates[0].get('Injury_Status')
        else:reason='Later role/pool exclusion; exact reason not archived';injury=None
        omissions.append({**r,'archived_reason':reason,'archived_injury_status':injury,
            'historical_projection_recoverable':False,'correction':'Fix fantasy-eligibility parser' if name in ['Travis Hunter','Hunter Luepke'] else 'Keep exclusion auditable; no hindsight reinsertion'})
with (OUT/'omitted_player_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(omissions[0]));w.writeheader();w.writerows(omissions)

# Snap counts distinguish actual offensive participation from a carry/target proxy.
cache=ROOT/'backtest_cache/calibration';cache.mkdir(parents=True,exist_ok=True)
if not (cache/'ids.parquet').exists() or not (cache/'snaps_2026.parquet').exists():
    import nflreadpy as nfl
    if not (cache/'ids.parquet').exists():nfl.load_ff_playerids().write_parquet(cache/'ids.parquet')
    if not (cache/'snaps_2026.parquet').exists():nfl.load_snap_counts([2026]).write_parquet(cache/'snaps_2026.parquet')
ids=pd.read_parquet(cache/'ids.parquet')
snaps=pd.read_parquet(cache/'snaps_2026.parquet')
def clean_id(x):
    if pd.isna(x):return None
    return str(x).removesuffix('.0')
ids['yahoo_key']=ids['yahoo_id'].map(clean_id)
mapping={}
for key,group in ids.dropna(subset=['yahoo_key','pfr_id']).groupby('yahoo_key'):
    values=group['pfr_id'].drop_duplicates().tolist()
    if len(values)==1:mapping['yahoo:'+key]=values[0]
participation=[]
for r in rows:
    if r['cohort']!='sleeper' or r['position']=='DEF':continue
    week=weeks[r['game_id']];pfr=mapping.get(r['player_key'])
    team=nb.sleeper.normalize_team(r['pregame_team'])
    found=snaps[(snaps['week']==week)&(snaps['pfr_player_id']==pfr)&(snaps['team']==team)] if pfr else snaps.iloc[:0]
    if len(found)==1:
        row=found.iloc[0];off=int(row['offense_snaps']);st=int(row['st_snaps']);status='offensive snaps' if off>0 else 'special-teams/defensive snaps only'
        # The actual game's team pair must agree with the pregame contest.
        model=jobs[r['game_id']]['payload'];teams={nb.sleeper.normalize_team(model['away']),nb.sleeper.normalize_team(model['home'])}
        if {team,nb.sleeper.normalize_team(row['opponent'])}!=teams:raise ValueError('Snap game mismatch')
    else:off=st=None;status='unknown: no unique snap record'
    participation.append(dict(game_id=r['game_id'],player_key=r['player_key'],player=r['player'],
        position=r['position'],role=r['role'],actual_yahoo_fp=r['actual_yahoo_fp'],
        offense_snaps=off,special_teams_snaps=st,participation_evidence=status))
with (OUT/'participation_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(participation[0]));w.writeheader();w.writerows(participation)
summary=dict(offensive_player_games=len(participation),matched_snaps=sum(r['offense_snaps'] is not None for r in participation),
    zero_with_offensive_snaps=sum(r['actual_yahoo_fp']==0 and (r['offense_snaps'] or 0)>0 for r in participation),
    zero_without_offensive_snaps_but_other_snaps=sum(r['actual_yahoo_fp']==0 and r['offense_snaps']==0 for r in participation),
    zero_participation_unknown=sum(r['actual_yahoo_fp']==0 and r['offense_snaps'] is None for r in participation),
    omitted_nonzero_records=len(omissions),historical_sleeper_raw_inputs_available=False,
    restriction='No omitted player is reinserted using postgame knowledge or current projections.')
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
