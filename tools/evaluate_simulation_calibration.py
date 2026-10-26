"""Out-of-time marginal scores and game-block correlation diagnostics."""
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline import notebook as nb, simulation_calibration as sc
BASE=ROOT/'backtest_output/exposure_2026';OUT=ROOT/'backtest_output/calibration_2026'
jobs={j['audit']['game_id']:j for p in OUT.glob('*.input.json') if (j:=json.loads(p.read_text()))}
grid=nb.normal_ppf((np.arange(16384)+.5)/16384)
weights=(2*np.arange(len(grid))-len(grid)+1)/len(grid)**2
def crps(samples,y):return float(np.mean(abs(samples-y))-weights@samples)
def baseline_cdf(y,p):
    if y<0:return 0.
    q=p.get('zero',0.)
    if y==0:return q
    sigma=math.sqrt(math.log1p(float(nb.conditional_lognormal_cv(p['cv'],q))**2))
    z=(math.log(y/(p['fp']/(1-q)))+.5*sigma*sigma)/sigma
    return q+(1-q)*float(nb.normal_cdf(z))
rows=[];pair_groups={};moment_errors=[]
for game,job in jobs.items():
    old=json.loads((BASE/f'{game}.input.json').read_text())['payload'];new=job['payload']
    oldcorr=nb.hurdle_score_correlation(sc.latent_matrix(old),np.array([p['cv'] for p in old['players']]),np.array([p.get('zero',0) for p in old['players']]))
    newcov=np.array(new['calibrated_covariance']).reshape(len(new['players']),-1)
    newstd=np.array([p['fp']*p['cv'] for p in new['players']]);newcorr=newcov/np.outer(newstd,newstd)
    for i,(p,q,y) in enumerate(zip(old['players'],new['players'],job['actuals'])):
        samples=nb.hurdle_transform(grid,p['fp'],p['cv'],p.get('zero',0))
        calibrated=sc.transform(grid,q['marginal']);m=q['marginal']
        analytic_mean=m['negative_probability']*m['negative_value']+(1-m['negative_probability']-m['zero_probability'])*m['positive_scale']
        moment_errors.append(abs(analytic_mean-p['fp']))
        z=nb.normal_ppf(np.array([.25,.9]))
        oldq=nb.hurdle_transform(z,p['fp'],p['cv'],p.get('zero',0));newq=sc.transform(z,m)
        rows.append(dict(game_id=game,player_key=job['player_keys'][i],position=p['pos'],role=p['role'],actual=y,
            baseline_zero=p.get('zero',0),calibrated_zero=m['zero_probability'],
            baseline_negative=0.,calibrated_negative=m['negative_probability'],
            baseline_zero_brier=(int(y==0)-p.get('zero',0))**2,
            calibrated_zero_brier=(int(y==0)-m['zero_probability'])**2,
            baseline_crps=crps(samples,y),calibrated_crps=crps(calibrated,y),
            baseline_p25_covered=int(y<=oldq[0]),calibrated_p25_covered=int(y<=newq[0]),
            baseline_p90_covered=int(y<=oldq[1]),calibrated_p90_covered=int(y<=newq[1])))
    # Same forecast-strength floor as the historical correlation calibration.
    for i,p in enumerate(old['players']):
        if p['fp']<2:continue
        for j in range(i+1,len(old['players'])):
            q=old['players'][j]
            if q['fp']<2:continue
            x=(job['actuals'][i]-p['fp'])/(p['fp']*p['cv'])
            y=(job['actuals'][j]-q['fp'])/(q['fp']*q['cv'])
            if nb.POSITION_ORDER[p['pos']]>nb.POSITION_ORDER[q['pos']]:p1,p2=q,p;x,y=y,x
            else:p1,p2=p,q
            relation=('same' if p1['team']==p2['team'] else 'opposing')+'_'+p1['pos']+'_'+p2['pos']
            pair_groups.setdefault(relation,[]).append((game,x,y,float(oldcorr[i,j]),float(newcorr[i,j])))
metrics={}
for label in ['ALL','QB','RB','WR','TE','DEF']:
    rr=[r for r in rows if label=='ALL' or r['position']==label]
    metrics[label]=dict(n=len(rr),observed_zeros=sum(r['actual']==0 for r in rr),observed_negatives=sum(r['actual']<0 for r in rr))
    for model in ['baseline','calibrated']:
        metrics[label][model]=dict(expected_zeros=sum(r[model+'_zero'] for r in rr),expected_negatives=sum(r[model+'_negative'] for r in rr),
            zero_brier=float(np.mean([r[model+'_zero_brier'] for r in rr])),crps=float(np.mean([r[model+'_crps'] for r in rr])),
            p25_coverage=float(np.mean([r[model+'_p25_covered'] for r in rr])),p90_coverage=float(np.mean([r[model+'_p90_covered'] for r in rr])))
def corr(m):
    n,sx,sy,sxx,syy,sxy=m[:6]
    vx=sxx-sx*sx/max(n,1);vy=syy-sy*sy/max(n,1)
    return (sxy-sx*sy/max(n,1))/math.sqrt(vx*vy) if vx>1e-12 and vy>1e-12 else float('nan')
groups=[];games=sorted(jobs);rng=np.random.default_rng(356)
draws=rng.integers(0,len(games),size=(3000,len(games)))
for relation,pairs in sorted(pair_groups.items()):
    aggregates=[]
    for game in games:
        rr=[r for r in pairs if r[0]==game]
        a=np.array([[1,x,y,x*x,y*y,x*y,old,new] for _,x,y,old,new in rr])
        aggregates.append(a.sum(axis=0) if len(rr) else np.zeros(8))
    a=np.array(aggregates);total=a.sum(axis=0);boots=np.array([corr(x) for x in a[draws].sum(axis=1)])
    finite=boots[np.isfinite(boots)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (float('nan'),float('nan'))
    target=total[6]/total[0]
    groups.append(dict(relationship=relation,pairs=int(total[0]),games=sum(x[0]>0 for x in a),
        empirical_residual_correlation=corr(total),game_bootstrap_low=float(lo),game_bootstrap_high=float(hi),
        baseline_target=target,calibrated_effective_target=total[7]/total[0],
        historical_target_inside_interval=bool(lo<=target<=hi)))
summary=dict(method='Expanding chronological fits; 24h outcome guard; 17 games / 432 player-game rows.',
    marginals=metrics,maximum_mean_preservation_error=max(moment_errors),
    maximum_score_correlation_adjustment=max(j['payload']['model']['psd_max_score_adjustment'] for j in jobs.values()),
    correlation_groups=len(groups),correlation_groups_target_inside_interval=sum(r['historical_target_inside_interval'] for r in groups),
    correlation_policy='Historical correlation targets retained, with attainable-range/PSD repair; diagnostics do not justify an unconstrained refit from 17 games.',
    limitations=['Correlations pool standardized forecast residuals for players projected at least 2 FP; pairs share games.',
        'Correlation intervals resample whole games, not pairs; exploratory across many relationships.',
        'OOF here excludes every evaluated game in the decision block from its calibration; the research design still followed inspection of this season.'])
score_games=sorted(jobs);bootstrap=np.random.default_rng(2026).integers(0,len(score_games),size=(30000,len(score_games)))
summary['paired_forecast_score_changes']={}
for metric in ['zero_brier','crps']:
    blocks=np.array([[sum(r['calibrated_'+metric]-r['baseline_'+metric] for r in rows if r['game_id']==g),
                      sum(r['game_id']==g for r in rows)] for g in score_games])
    sampled=blocks[bootstrap].sum(axis=1);changes=sampled[:,0]/sampled[:,1]
    summary['paired_forecast_score_changes'][metric]=dict(calibrated_minus_baseline=float(blocks[:,0].sum()/blocks[:,1].sum()),
        game_bootstrap_95=list(map(float,np.quantile(changes,[.025,.975]))),lower_is_better=True)
(OUT/'simulation_evaluation.json').write_text(json.dumps(summary,indent=2))
for name,rr in [('calibration_player_scores.csv',rows),('correlation_validation.csv',groups)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
print(json.dumps(summary,indent=2))
