"""Research calibration of pregame-eligible scoring marginals.

Fit only earlier finalized games. This is opt-in payload generation, not a
production-default change. The positive tail remains a lognormal approximation.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
import pandas as pd
from pipeline import notebook as nb

PRIOR_STRENGTH = 20.0  # fixed before evaluation; not selected on portfolio scores

def bucket(position, depth):
    return f'{position}:{min(max(int(depth),1),4)}'

def fit(rows, cutoff, prior_strength=PRIOR_STRENGTH):
    if not math.isfinite(prior_strength) or prior_strength<=0:raise ValueError('Prior strength must be positive')
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:raise ValueError('Calibration cutoff must be timezone aware')
    eligible=[r for r in rows if pd.Timestamp(r['kickoff_utc'])+pd.Timedelta(hours=24)<cutoff]
    if len(eligible)!=len(rows):raise ValueError('Training outcome is not safely before cutoff')
    if not rows:raise ValueError('Earlier training observations are required')
    negative=[r['actual_yahoo_fp'] for r in rows if r['actual_yahoo_fp']<0]
    global_neg=len(negative)/len(rows)
    cells={}
    for position in nb.VALID_POSITIONS:
        posrows=[r for r in rows if r['position']==position]
        neg=[r['actual_yahoo_fp'] for r in posrows if r['actual_yahoo_fp']<0]
        neg_prob=(len(neg)+prior_strength*global_neg)/(len(posrows)+prior_strength)
        neg_value=float(np.mean(neg or negative or [-1.0]))
        for depth in range(1,5):
            cell=[r for r in posrows if min(int(r['depth']),4)==depth]
            p0=(sum(r['actual_yahoo_fp']==0 for r in cell)+prior_strength*nb.ZERO_RATE[position][depth])/(len(cell)+prior_strength)
            # Positive-shape residuals must come from the same Sleeper forecast era.
            positives=[r for r in cell if r['cohort']=='sleeper' and r['actual_yahoo_fp']>0 and r['projected_fp']>0]
            logs=np.log([r['actual_yahoo_fp']/r['projected_fp'] for r in positives])
            prior_sigma2=math.log1p(float(nb.conditional_lognormal_cv(nb.CALIBRATED_CV[position][depth],nb.ZERO_RATE[position][depth]))**2)
            sigma2=(float(np.sum((logs-logs.mean())**2)) if len(logs)>1 else 0.0)+prior_strength*prior_sigma2
            sigma2/=max(len(logs)-1,0)+prior_strength
            if p0+neg_prob>=1:raise ValueError('No positive probability remains')
            cells[bucket(position,depth)]=dict(zero_probability=p0,negative_probability=neg_prob,
                negative_value=neg_value,positive_sigma=math.sqrt(max(sigma2,1e-8)),
                training_rows=len(cell),positive_shape_rows=len(positives))
    training_games=sorted({r['game_id'] for r in rows})
    return dict(schema=1,family='calibrated_signed_hurdle_lognormal_v1',cutoff_utc=cutoff.isoformat(),
        training_games=training_games,training_rows=len(rows),prior_strength=prior_strength,cells=cells,
        last_training_kickoff=max(pd.Timestamp(r['kickoff_utc']) for r in rows).isoformat(),
        limitations=['Zero probability is unconditional on opportunity within the archived eligible pool.',
            'Zero outcomes do not identify inactive versus active-but-scoreless participation.',
            'Positive scores remain continuous and team scoring totals are not constrained.',
            'Negative scores use a shrunken probability and a position-pooled negative atom.',
            'Legacy games supply sign frequencies only; positive-tail residuals use earlier Sleeper games.'])

def marginal(mean, cell):
    p0=cell['zero_probability'];pn=cell['negative_probability'];negative=cell['negative_value']
    pp=1-p0-pn
    sigma=cell['positive_sigma']
    if not all(math.isfinite(x) for x in [mean,p0,pn,negative,sigma]) or mean<=0 or p0<0 or pn<0 or pp<=0 or negative>=0 or sigma<=0:
        raise ValueError('Invalid signed marginal')
    scale=(mean-pn*negative)/pp
    second=pn*negative**2+pp*scale**2*math.exp(sigma*sigma)
    variance=max(second-mean*mean,0)
    return dict(zero_probability=p0,negative_probability=pn,negative_value=negative,
        positive_sigma=sigma,positive_scale=scale),math.sqrt(variance)/mean

def transform(latent, m):
    u=nb.normal_cdf(latent);pn=m['negative_probability'];p0=m['zero_probability'];pp=1-pn-p0
    q=np.clip((u-pn-p0)/pp,1e-300,1-1e-16)
    positive=m['positive_scale']*np.exp(nb.normal_ppf(q)*m['positive_sigma']-.5*m['positive_sigma']**2)
    return np.where(u<pn,m['negative_value'],np.where(u<pn+p0,0.,positive))

def cdf(value,m,left=False):
    """CDF, including left limits for randomized PIT on atoms."""
    pn=m['negative_probability'];p0=m['zero_probability'];negative=m['negative_value']
    if value<negative or (left and value==negative):return 0.
    if value<0 or (left and value==0):return pn
    if value==0:return pn+p0
    z=(math.log(value/m['positive_scale'])+.5*m['positive_sigma']**2)/m['positive_sigma']
    return pn+p0+(1-pn-p0)*float(nb.normal_cdf(z))

def coefficients(mean,m):
    grid=np.linspace(-9.,9.,20001);spacing=grid[1]-grid[0]
    weighted=transform(grid,m)/mean*np.exp(-.5*grid**2)/math.sqrt(2*math.pi)
    def integral(values):return spacing*(values.sum()-.5*(values[0]+values[-1]))
    out=np.empty(25);prev=np.ones_like(grid);curr=grid.copy()
    out[0]=integral(weighted);out[1]=integral(weighted*curr)
    for k in range(1,24):
        prev,curr=curr,grid*curr-k*prev
        out[k+1]=integral(weighted*curr)
    return out

def latent_matrix(payload):
    n=len(payload['players']);matrix=np.eye(n);ix=np.triu_indices(n,1)
    matrix[ix]=payload['latent'];matrix[(ix[1],ix[0])]=payload['latent']
    return matrix

def calibrated_payload(payload, artifact):
    if pd.Timestamp(artifact['cutoff_utc'])>pd.Timestamp(payload['kickoff_utc']):raise ValueError('Calibration is after game kickoff')
    if payload['game_id'] in artifact['training_games']:raise ValueError('Game leaked into calibration')
    out=deepcopy(payload);players=out['players'];n=len(players)
    oldcv=np.array([p['cv'] for p in players]);oldzero=np.array([p.get('zero',0) for p in players])
    target=nb.hurdle_score_correlation(latent_matrix(payload),oldcv,oldzero)
    coeff=[];cv=[]
    for p in players:
        m,newcv=marginal(p['fp'],artifact['cells'][bucket(p['pos'],p['depth'])])
        p['marginal']=m;p['cv']=newcv;p['zero']=m['zero_probability']
        coeff.append(coefficients(p['fp'],m));cv.append(newcv)
    cv=np.array(cv);coeff=np.array(coeff)
    limit=np.full((n,n),.95)  # keep truncated-series inversion away from singular extremes
    low=nb._hurdle_correlation_from_latent(-limit,coeff,cv)
    high=nb._hurdle_correlation_from_latent(limit,coeff,cv)
    feasible=np.clip(target,low+1e-6,high-1e-6);np.fill_diagonal(feasible,1)
    lower=-limit;upper=limit
    for _ in range(45):
        middle=(lower+upper)/2
        implied=nb._hurdle_correlation_from_latent(middle,coeff,cv)
        lower=np.where(implied<feasible,middle,lower);upper=np.where(implied<feasible,upper,middle)
    latent=(lower+upper)/2
    np.fill_diagonal(latent,1.0)
    latent=nb.repair_correlation_matrix(latent)
    effective=nb._hurdle_correlation_from_latent(latent,coeff,cv)
    cov=np.outer([p['fp'] for p in players],[p['fp'] for p in players])*np.outer(cv,cv)*effective
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min()< -1e-6:
        raise ValueError('Invalid calibrated covariance')
    out['latent']=latent[np.triu_indices(n,1)].tolist()
    out['calibrated_covariance']=cov.ravel().tolist()
    out['model']=dict(family=artifact['family'],calibration_cutoff=artifact['cutoff_utc'],
        calibration_training_games=len(artifact['training_games']),
        psd_max_score_adjustment=float(np.max(abs(effective-target))),
        limitations=artifact['limitations'])
    out.pop('reference',None)
    return out
