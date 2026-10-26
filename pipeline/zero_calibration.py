"""Validated default zero-only calibration; no signed-score or tail refit."""
from datetime import datetime,timedelta
import json
import math
from pathlib import Path

def load(path,legacy_cv,legacy_zero):
    artifact=json.loads(Path(path).read_text())
    if artifact.get('schema')!=1 or artifact.get('family')!='eligible_pool_zero_only_v1':
        raise ValueError('Unsupported zero-score calibration artifact')
    cutoff=datetime.fromisoformat(artifact['cutoff_utc'])
    last=datetime.fromisoformat(artifact['last_training_kickoff'])
    if cutoff.tzinfo is None or last.tzinfo is None or last+timedelta(hours=24)>=cutoff:
        raise ValueError('Zero calibration lacks a valid training cutoff')
    cv={};zero={}
    for pos in legacy_cv:
        cv[pos]={};zero[pos]={}
        for depth in range(1,5):
            q=float(artifact['zero'][pos][str(depth)]);c=float(artifact['cv'][pos][str(depth)])
            if not math.isfinite(q) or not math.isfinite(c) or not 0<=q<1 or c<=0:
                raise ValueError('Invalid zero-score probability or CV')
            old_shape=legacy_cv[pos][depth]**2*(1-legacy_zero[pos][depth])-legacy_zero[pos][depth]
            new_shape=c*c*(1-q)-q
            if new_shape<=0 or abs(old_shape-new_shape)>1e-8:
                raise ValueError('Zero-only calibration changed positive conditional shape')
            cv[pos][depth]=c;zero[pos][depth]=q
    return artifact,cv,zero
