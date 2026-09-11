"""Immutable pregame snapshots and leakage-resistant forecast evaluation."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def clean(value):
    if is_dataclass(value):
        return clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def capture(prepared, cfg, snapshot_id, showdown_payloads):
    from pipeline import notebook as nb
    root=Path(__file__).resolve().parents[1]
    files=sorted((root/'pipeline').glob('*.py'))+[root/'site/showdown-worker.js']
    digest=hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    rows=[]
    for row in prepared['players'].to_dict('records'):
        identity = row.get('Yahoo ID')
        key = ('yahoo:' + str(int(identity))) if pd.notna(identity) else (
            row['Team'] + ':' + nb.normalize_person_name(row['Name']))
        mean=float(row['Projected_FP']);cv=nb._calibrated_cv(row['Position'],row['Depth_Rank'])
        sigma=math.sqrt(math.log1p(cv*cv))
        quantile=lambda z: mean*math.exp(z*sigma-.5*sigma*sigma)
        rows.append(dict(game_id=str(row['Game ID']),player_key=key,player=row['Name'],
            team=row['Team'],position=row['Position'],kickoff_utc=row['Game Time'],
            expected_fp=mean,baseline_fp=row.get('Role_Adjusted_Baseline_FP'),
            p25=quantile(-.6744897501960817),p90=quantile(1.2815515655446004),
            source=row['Projection_Source'],salary=row['Salary']))
    return clean(dict(schema=1,snapshot_id=snapshot_id,
        captured_utc=prepared.get('inputs_captured_utc',datetime.now(timezone.utc).isoformat()),
        code_sha256=digest.hexdigest(),settings=asdict(cfg),predictions=rows,
        projection_model=prepared.get('projection_model',{}),
        role_report=json.loads(prepared.get('nflverse_report',pd.DataFrame()).to_json(orient='records',date_format='iso')),
        showdown_models=showdown_payloads,
        raw_inputs=prepared.get('raw_inputs',{})))


def _bytes(value):
    return json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _immutable(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes()!=data:
            raise ValueError(f'Archive collision: {path.name}')


def write(snapshot, directory):
    """Deduplicate raw feeds by content hash, retaining every forecast snapshot."""
    directory=Path(directory)
    manifest=dict(snapshot)
    refs={}
    for name, payload in manifest.pop('raw_inputs',{}).items():
        raw=_bytes(payload);digest=hashlib.sha256(raw).hexdigest()
        relative=f'objects/{digest}.json.gz'
        _immutable(directory/relative,gzip.compress(raw,mtime=0))
        refs[name]=relative
    manifest['raw_input_refs']=refs
    raw=_bytes(manifest);digest=hashlib.sha256(raw).hexdigest()
    path=directory/'snapshots'/f'{digest}.json.gz'
    _immutable(path,gzip.compress(raw,mtime=0))
    return path


def evaluate(snapshots, actuals):
    """One latest pregame forecast per game/player; no repeated-run pseudo-samples.

    Actuals need game_id, player_key, actual_fp, realized_at_utc. Identity matching
    is explicit; missing actuals never become zero. Manual overrides are excluded.
    """
    required={'game_id','player_key','actual_fp','realized_at_utc'}
    if not required <= set(actuals.columns):
        raise ValueError('Actuals need '+', '.join(sorted(required)))
    actuals=actuals.copy()
    actuals['game_id']=actuals.game_id.astype(str)
    if actuals.duplicated(['game_id','player_key']).any():
        raise ValueError('Duplicate actual game/player identities')
    lookup=actuals.set_index(['game_id','player_key']).to_dict('index')
    latest={};rejected=0
    for snapshot in snapshots:
        captured=pd.Timestamp(snapshot['captured_utc'])
        if captured.tzinfo is None: raise ValueError('Snapshot timestamp must include timezone')
        for row in snapshot['predictions']:
            kickoff=pd.Timestamp(row['kickoff_utc'])
            if kickoff.tzinfo is None: raise ValueError('Kickoff timestamp must include timezone')
            if captured>=kickoff or row['source']=='manual override':
                rejected+=1;continue
            key=(str(row['game_id']),row['player_key'])
            if key not in latest or captured>latest[key][0]: latest[key]=(captured,row)
    pairs=[]
    for key,(_, row) in latest.items():
        actual=lookup.get(key)
        if actual is None: continue
        finished=pd.Timestamp(actual['realized_at_utc'])
        if finished.tzinfo is None or finished<=pd.Timestamp(row['kickoff_utc']):
            raise ValueError('Actual result timestamp must be after kickoff')
        value=float(actual['actual_fp'])
        if not math.isfinite(value): raise ValueError('Non-finite actual points')
        pairs.append((row,value))
    def metrics(items):
        if not items:return {'n':0}
        errors=np.array([row['expected_fp']-value for row,value in items])
        matched=[(row,value) for row,value in items if row.get('baseline_fp') is not None]
        result=dict(n=len(items),mae=float(np.abs(errors).mean()),
            rmse=float(np.sqrt(np.square(errors).mean())),bias=float(errors.mean()),
            below_p25=float(np.mean([value<row['p25'] for row,value in items])),
            below_p90=float(np.mean([value<row['p90'] for row,value in items])),
            zero_or_negative_rate=float(np.mean([value<=0 for _,value in items])),
            baseline_n=len(matched))
        if matched:
            result['baseline_mae']=float(np.mean([abs(row['baseline_fp']-v) for row,v in matched]))
            result['paired_model_mae']=float(np.mean([abs(row['expected_fp']-v) for row,v in matched]))
        return result
    return dict(schema=1,selection='latest captured before kickoff; unique game/player',
        rejected_postkickoff_or_manual=rejected,unmatched_predictions=len(latest)-len(pairs),
        overall=metrics(pairs),by_position={p:metrics([(r,v) for r,v in pairs if r['position']==p])
            for p in sorted({r['position'] for r,_ in pairs})},
        limitations=['Empirical evaluation of archived forecasts; does not refit weights.',
          'Nominal below-P25/P90 rates are 25%/90%; discrete scores can produce ties.'])

