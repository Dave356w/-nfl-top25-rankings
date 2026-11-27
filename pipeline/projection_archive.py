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


def _optional_float(value):
    try:
        value=float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _optional_text(value):
    return None if value is None or (isinstance(value,float) and not math.isfinite(value)) else str(value)


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
            source=row['Projection_Source'],salary=row['Salary'],
            # Display-only market-implied points from player props (null when the
            # books do not price him or the market feed is off), kept so the
            # market can be graded beside the blend it never feeds.
            market_fp=_optional_float(row.get('Market_FP')),
            market_quality=_optional_text(row.get('Market_Quality'))))
    return clean(dict(schema=1,snapshot_id=snapshot_id,
        captured_utc=prepared.get('inputs_captured_utc',datetime.now(timezone.utc).isoformat()),
        code_sha256=digest.hexdigest(),settings=asdict(cfg),predictions=rows,
        projection_model=prepared.get('projection_model',{}),
        availability_removed=json.loads(prepared.get('availability_removed',pd.DataFrame()).to_json(orient='records',date_format='iso')),
        role_filters=dict(backup_qbs_removed=prepared.get('backup_qbs_removed',[]),
                          excluded=prepared.get('excluded',[])),
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
        # Compare content, not compressed bytes: the same JSON gzips to
        # different bytes under different zlib builds (a runner image update),
        # and that must not fail a publish.
        existing=path.read_bytes()
        if existing!=data and gzip.decompress(existing)!=gzip.decompress(data):
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


def load_exclusions(path):
    """Snapshot IDs that must never be graded, mapped to the reason (missing file: none)."""
    path=Path(path)
    if not path.exists(): return {}
    excluded=json.loads(path.read_text()).get('excluded') or {}
    if not isinstance(excluded,dict) or not all(str(r).strip() for r in excluded.values()):
        raise ValueError(f'{path}: "excluded" must map snapshot IDs to non-empty reasons')
    return {str(k):str(v) for k,v in excluded.items()}


def evaluate(snapshots, actuals, excluded=None, market_backfill=None):
    """One latest pregame forecast per game/player; no repeated-run pseudo-samples.

    Actuals need game_id, player_key, actual_fp, realized_at_utc. Identity matching
    is explicit; missing actuals never become zero. Manual overrides are excluded,
    as is every snapshot whose ID is in ``excluded``: its forecasts are dropped
    before selection, so an earlier valid forecast is graded in its place.

    The market (display-only prop-implied points) is graded the same way: its
    latest pregame value per game/player, from the snapshots' ``market_fp`` and
    from ``market_backfill`` rows (snapshot_id, captured_utc, game_id,
    player_key, kickoff_utc, market_fp, market_quality) for runs archived before
    snapshots carried it. It is paired with the blend forecast graded above.
    """
    excluded=set(excluded or ())
    skipped=sorted({s['snapshot_id'] for s in snapshots if s.get('snapshot_id') in excluded})
    snapshots=[s for s in snapshots if s.get('snapshot_id') not in excluded]
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
    market={}
    def observe_market(captured,kickoff,key,value,quality,origin):
        if value is None or not math.isfinite(float(value)) or captured>=kickoff:return
        if key not in market or captured>market[key][0]:
            market[key]=(captured,float(value),quality,origin)
    for snapshot in snapshots:
        captured=pd.Timestamp(snapshot['captured_utc'])
        for row in snapshot['predictions']:
            if row.get('market_fp') is None or row['source']=='manual override':continue
            observe_market(captured,pd.Timestamp(row['kickoff_utc']),(str(row['game_id']),row['player_key']),
                           row['market_fp'],row.get('market_quality'),'snapshot')
    backfill_rows=0
    if market_backfill is not None and len(market_backfill):
        for row in market_backfill.to_dict('records'):
            if str(row['snapshot_id']) in excluded:continue
            captured=pd.Timestamp(row['captured_utc']);kickoff=pd.Timestamp(row['kickoff_utc'])
            if captured.tzinfo is None or kickoff.tzinfo is None:
                raise ValueError('Market backfill timestamps must include timezone')
            quality=row.get('market_quality')
            quality=None if quality is None or (isinstance(quality,float) and not math.isfinite(quality)) else str(quality)
            observe_market(captured,kickoff,(str(row['game_id']),row['player_key']),
                           row['market_fp'],quality,'backfill')
            backfill_rows+=1
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
    market_pairs=[]
    for row,value in pairs:
        observed=market.get((str(row['game_id']),row['player_key']))
        if observed is not None:market_pairs.append((row,value,observed))
    def market_metrics(items):
        if not items:return {'n':0}
        actual=np.array([v for _,v,_ in items]);blend=np.array([r['expected_fp'] for r,_,_ in items])
        price=np.array([o[1] for _,_,o in items]);half=(blend+price)/2
        gap=price-blend;residual=actual-blend
        result=dict(n=len(items),
            from_backfill=int(sum(o[3]=='backfill' for _,_,o in items)),
            market_mae=float(np.abs(price-actual).mean()),market_bias=float((price-actual).mean()),
            blend_mae=float(np.abs(blend-actual).mean()),blend_bias=float((blend-actual).mean()),
            half_market_mae=float(np.abs(half-actual).mean()),
            market_blend_corr=float(np.corrcoef(price,blend)[0,1]) if len(items)>2 else None)
        # Slope of (actual - blend) on (market - blend): ~0 means the market adds
        # nothing beyond the blend, ~1 means its disagreement is fully informative.
        result['gap_residual_slope']=(float(np.dot(gap-gap.mean(),residual-residual.mean())/
            np.square(gap-gap.mean()).sum()) if len(items)>2 and np.square(gap-gap.mean()).sum()>0 else None)
        if len(items)>2:
            draws=np.random.default_rng(0).integers(0,len(items),(2000,len(items)))
            gain=np.abs(half-actual)[draws].mean(1)-np.abs(blend-actual)[draws].mean(1)
            result['half_minus_blend_mae_ci95']=[float(np.percentile(gain,2.5)),float(np.percentile(gain,97.5))]
        return result
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
        excluded_snapshots=skipped,rejected_postkickoff_or_manual=rejected,unmatched_predictions=len(latest)-len(pairs),
        overall=metrics(pairs),by_position={p:metrics([(r,v) for r,v in pairs if r['position']==p])
            for p in sorted({r['position'] for r,_ in pairs})},
        market=dict(backfill_rows=backfill_rows,overall=market_metrics(market_pairs),
            by_position={p:market_metrics([m for m in market_pairs if m[0]['position']==p])
                for p in sorted({m[0]['position'] for m in market_pairs})},
            by_quality={q:market_metrics([m for m in market_pairs if (m[2][2] or 'unknown')==q])
                for q in sorted({m[2][2] or 'unknown' for m in market_pairs})}),
        limitations=['Empirical evaluation of archived forecasts; does not refit weights.',
          'Nominal below-P25/P90 rates are 25%/90%; discrete scores can produce ties.',
          'Market rows are display-only prop prices paired with the graded blend; '
          'half_market is a diagnostic 50/50 mix, not a live model. Players in one '
          'game share outcomes, so the bootstrap interval is optimistic.'])

