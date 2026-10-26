"""Build the default zero-only calibration from already-settled eligible pools."""
import argparse
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline import notebook as nb
from pipeline import simulation_calibration as sc

def isotonic(values,weights):
    blocks=[]
    for i,(value,weight) in enumerate(zip(values,weights)):
        blocks.append([i,i,value*weight,weight])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    out=list(values)
    for start,end,total,weight in blocks:
        for i in range(start,end+1):out[i]=total/weight
    return out

def build(source,cutoff):
    data=list(csv.DictReader(Path(source).open()))
    rows=[]
    for row in data:
        if pd.Timestamp(row['kickoff_utc'])+pd.Timedelta(hours=24)>=pd.Timestamp(cutoff):continue
        for k in ['actual_yahoo_fp','projected_fp']:row[k]=float(row[k])
        row['depth']=int(row['depth']);rows.append(row)
    fitted=sc.fit(rows,cutoff)
    legacy_cv=getattr(nb,'LEGACY_CALIBRATED_CV',nb.CALIBRATED_CV)
    legacy_zero=getattr(nb,'LEGACY_ZERO_RATE',nb.ZERO_RATE)
    zero={};cv={}
    for pos in nb.VALID_POSITIONS:
        cells=[fitted['cells'][sc.bucket(pos,d)] for d in range(1,5)]
        rates=[c['zero_probability'] for c in cells]
        rates=[rates[0]]*4 if pos=='DEF' else isotonic(rates,[c['training_rows']+fitted['prior_strength'] for c in cells])
        zero[pos]={str(d):rates[d-1] for d in range(1,5)};cv[pos]={}
        for d in range(1,5):
            conditional2=legacy_cv[pos][d]**2*(1-legacy_zero[pos][d])-legacy_zero[pos][d]
            cv[pos][str(d)]=math.sqrt((conditional2+rates[d-1])/(1-rates[d-1]))
    return dict(schema=1,family='eligible_pool_zero_only_v1',cutoff_utc=fitted['cutoff_utc'],
        last_training_kickoff=fitted['last_training_kickoff'],training_games=fitted['training_games'],
        training_rows=len(rows),prior_strength=fitted['prior_strength'],zero=zero,cv=cv,
        source='Exact archived pregame eligible pools matched to Yahoo settled points.',
        method='Unconditional zero probabilities by position/depth, beta shrinkage and weighted monotonic pooling. Preserve legacy positive conditional shape and unconditional means; recalculate total CV.',
        limitations=['This is zero-only calibration; signed-score and positive-tail refits remain research-only.',
            'Negative outcomes remain unsupported by this production distribution.',
            'Future validation is required; portfolio ROI has not been established.'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(ROOT/'backtest_output/exposure_2026/player_game_outcomes.csv'))
    parser.add_argument('--cutoff',default=datetime.now(timezone.utc).isoformat())
    parser.add_argument('--output',default=str(ROOT/'model/zero_score_calibration.json'))
    a=parser.parse_args();artifact=build(a.source,a.cutoff)
    Path(a.output).write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps({k:artifact[k] for k in ['family','cutoff_utc','training_rows','training_games']},indent=2))
