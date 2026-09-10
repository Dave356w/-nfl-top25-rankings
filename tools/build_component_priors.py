#!/usr/bin/env python3
"""Build conditional-on-appearance component priors using time-ordered nflverse data.

Tune recency weighting before a holdout season, then approve each position only
if total-point MAE improves by 2% over an eight-appearance rolling mean on at
least 200 holdout rows. No Yahoo prices or archived market forecasts are used
in this comparison; it does not validate the eventual market/prior blend.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.calibrate_rb_correlation import load_weekly

COMPONENTS = ('passing_yards','passing_touchdowns','interceptions','rushing_yards',
              'receiving_yards','receptions','any_touchdowns','fumbles_lost')
COEFFICIENTS = np.array([.04,4,-1,.1,.1,.5,6,-2])
ALPHAS = (.15,.3,.5)


def prepare(weekly):
    x = weekly[weekly.position.isin(['QB','RB','WR','TE']) & weekly.season_type.eq('REG')].copy()
    x['passing_touchdowns'] = x['passing_tds']
    x['interceptions'] = x['passing_interceptions']
    x['any_touchdowns'] = x['rushing_tds'] + x['receiving_tds']
    x['fumbles_lost'] = x[['sack_fumbles_lost','rushing_fumbles_lost','receiving_fumbles_lost']].sum(axis=1, min_count=3)
    # Unknown stat cells are not silently made into zeros.
    x = x.dropna(subset=list(COMPONENTS) + ['player_id','player_display_name'])
    if x.duplicated(['player_id','season','week']).any():
        raise ValueError('Duplicate weekly player identities must be resolved before fitting.')
    return x.sort_values(['player_id','season','week'])


def weighted(values, alpha):
    weights = (1-alpha) ** np.arange(len(values)-1,-1,-1)
    return np.average(values,axis=0,weights=weights)


def build(weekly, holdout, as_of):
    stamp = pd.Timestamp(as_of)
    if stamp.tzinfo is None:
        raise ValueError('as_of must include a timezone')
    # Full-season files cannot safely train an artifact before that season ends.
    if stamp < pd.Timestamp(f'{holdout+1}-03-01',tz='UTC'):
        raise ValueError('The holdout season must have ended before as_of.')
    x = prepare(weekly[weekly.season.le(holdout)])
    records, latest = [], []
    for player_id, group in x.groupby('player_id',sort=False):
        history = []
        for row in group.itertuples(index=False):
            values = np.array([getattr(row,c) for c in COMPONENTS],float)
            recent = [v for year,v in history if year >= row.season-1][-8:]
            if len(recent) == 8:
                recent = np.array(recent)
                actual = float(values @ COEFFICIENTS)
                records.append(dict(position=row.position,season=int(row.season),
                    baseline_error=abs(float(recent.mean(axis=0) @ COEFFICIENTS)-actual),
                    errors=[abs(float(weighted(recent,a) @ COEFFICIENTS)-actual) for a in ALPHAS]))
            history.append((row.season,values))
        row = group.iloc[-1]
        recent = [v for year,v in history if year >= holdout-1][-8:]
        if len(recent) == 8 and int(row.season) == holdout:
            latest.append((row,recent))
    validation = {}
    for pos in ('QB','RB','WR','TE'):
        train = [r for r in records if r['position']==pos and r['season']<holdout]
        test = [r for r in records if r['position']==pos and r['season']==holdout]
        if not train or not test:
            validation[pos]={'approved':False,'n':len(test),'reason':'insufficient history'}
            continue
        chosen = int(np.argmin(np.mean([r['errors'] for r in train],axis=0)))
        baseline = float(np.mean([r['baseline_error'] for r in test]))
        candidate = float(np.mean([r['errors'][chosen] for r in test]))
        validation[pos] = dict(approved=len(test)>=200 and candidate<=.98*baseline,
            n=len(test),alpha=ALPHAS[chosen],baseline_mae=baseline,candidate_mae=candidate,
            improvement_pct=100*(baseline-candidate)/baseline)
    priors=[]
    for row, recent in latest:
        result = validation[row.position]
        if not result['approved']:
            continue
        means=weighted(np.array(recent),result['alpha'])
        priors.append(dict(player_id=row.player_id,player=str(row.player_display_name),
            position=str(row.position),last_season=int(row.season),last_week=int(row.week),
            appearances=8,stat_means=dict(zip(COMPONENTS,map(float,means)))))
    return dict(schema=1,as_of_utc=stamp.isoformat(),trained_through_season=holdout,
        holdout_season=holdout,scoring='yahoo',validation=validation,priors=priors,
        limitations=['Conditional on a recorded weekly appearance; not an availability model.',
          'Validation compares historical point priors, not market blends or Yahoo salary priors.',
          'Only earlier weekly observations enter each prediction; holdout does not select alpha.',
          'Retrospective nflverse files may contain later corrections; original publication timestamps are not available.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout',type=int,default=2025)
    parser.add_argument('--start',type=int,default=2021)
    parser.add_argument('--as-of',default=datetime.now(timezone.utc).isoformat())
    parser.add_argument('--output',default='model/component_priors.json')
    args=parser.parse_args()
    artifact=build(load_weekly(list(range(args.start,args.holdout+1))),args.holdout,args.as_of)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    print(json.dumps(artifact['validation'],indent=2));print('Supported priors:',len(artifact['priors']))

if __name__=='__main__': main()

