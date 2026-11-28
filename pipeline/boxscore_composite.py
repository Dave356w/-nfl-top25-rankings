"""NFL box-score composite W/L model — v1.7 (production).

Ported into this repo on 2026-11-28 from the Colab notebook NFL_C_B_MODEL.ipynb
(the GitHub NFL-MODEL repo was disabled). The model code is unchanged; only the
output folder is configurable (BOXSCORE_OUTPUT_ROOT, default
boxscore_cache/nfl_boxscore_output_v1_7) so caches stay out of git. The frozen
2026 recipe and forward ledger from the Drive run are committed under
model/boxscore_composite/ and copied into the output folder by
tools/build_boxscore_composite.py. The margin research page (ats.html) uses
this model in place of the Sleeper points model.

Paste the entire file into ONE Colab cell; or python nfl_boxscore_composite_v1_7.py.
Offline checks: python nfl_boxscore_composite_v1_7.py --self-test

v1.7 (NEW experiment: new REVISION and OUTPUT_ROOT; the v1.6 folder, frozen
recipe and forward ledger are left untouched): opponent-adjusted availability.
  * New family 'rates_core_adj_avail_cs' = the opponent-adjusted core stats of
    'rates_core_adj' (v1.3) plus the current-season availability layer of
    'rates_core_avail_cs' (v1.6). The unadjusted v1.6 winner stays in the
    candidate set, so walk-forward selection decides whether adjustment helps.
  * Candidate families trimmed to the two above (24 candidates instead of 36);
    rates_core and rates_core_avail lost to rates_core_avail_cs in v1.6.
  * Manifest limitations and board footer describe opponent adjustment
    according to the selected recipe.

v1.6.5 (speed only; same REVISION, signature, recipe, ledger and outputs):
computed availability features for completed seasons are cached per season
under OUTPUT_ROOT/cache, so each run rebuilds only the current season's
team-weeks. A season's cache key hashes the config signature and every input
that season can read: its schedule rows and the injury, roster, snap and QB
rows for that season and the MAX_HISTORY_SEASONS before it. Any upstream or
config change therefore rebuilds that season. Per-season audit counts are
stored beside each cache file, so the printed roster-membership totals still
cover all seasons.

v1.6.4 (reporting only; same REVISION, signature, recipe and ledger): realized
pick records in matched bands, game-card style. Each source "picks" the side it
makes the favorite; pick confidence is binned on the SAME edges for model and
market (PICK_BAND_EDGES: 50-55, 55-60, ... 80%+). For every band, the board and
printout show each source's W-L record, average pick probability and realized
pick win rate (Wilson 95%). Every game card adds a line with the model's record
in the model's band for that game and the market's record in the market's band.
Records use held-out seasons plus earlier weeks of the current season, all
walk-forward reconstructions; the current week never feeds its own records.
The v1.6.3 home-win calibration is still saved to calibration_bands.csv.

v1.6.3 (reporting only; same REVISION, signature, recipe and ledger): calibration
in matched bands. Every held-out game is binned twice on the SAME home-win
probability edges (CAL_BAND_EDGES): once by the model's forecast, once by the
market's. Each band reports games, mean forecast and realized home-win rate with
a Wilson 95% interval; the board shows a reliability chart and table.

v1.6.2 (storage only; same REVISION, signature, recipe and ledger): in Colab,
OUTPUT_ROOT lives on Google Drive (MyDrive/nfl_boxscore/<OUTPUT_NAME>),
so the frozen recipe, forward ledger, caches and run folders survive sessions.
Drive is mounted on first run (one authorization prompt). A local folder left
by an earlier session in this runtime is copied over once, never overwriting
anything already on Drive; forward-ledger records are merged by experiment and
game. Outside Colab, or with USE_GOOGLE_DRIVE=False, the local folder is used.

v1.6.1 (reporting only; same REVISION, config signature, frozen recipe and
forward ledger as v1.6, so no restart): adds a market-blend diagnostic and
fixes two stale board labels. Predictions are unchanged.
  * Market-blend check: logistic regression of home W/L on the spread's
    log-odds and the composite's log-odds. (a) Pooled held-out fit with a
    game-bootstrap interval for the composite weight: is it reliably nonzero?
    (b) Chronological blend: weights fitted on earlier walk-forward seasons
    (same recipe), applied to each held-out season, scored against the market.
    Earlier seasons also informed recipe selection, so (b) is mildly optimistic.

v1.6: availability fixes found in the v1.5 run (NEW experiment: new REVISION
and OUTPUT_ROOT; earlier folders and forward ledgers are left untouched).
  * Bye weeks: weekly rosters have no row for a team on bye, so the game after
    every bye had no "prior week" roster and membership/status silently
    switched off (about 220 team-weeks). The reference roster is now the
    team's MOST RECENT roster week before the forecast week.
  * QB projection: only players listed at QB on the reference roster can be
    projected. If none of them has dropbacks for the team (e.g. starter out,
    backup newly signed), the projection is the league backup prior and the
    board says so, instead of naming a gadget passer with one trick-play throw.
  * New candidate family 'rates_core_avail_cs': identical to rates_core_avail,
    except unit unavailability uses only current-season games once a team has
    CS_MIN_GAMES of them, so last season's finale (and offseason departures in
    it) stops counting once this season's lineup is visible.

v1.5: availability fix and labelling (NEW experiment: new REVISION and OUTPUT_ROOT;
the v1.4 folder, frozen recipe and forward ledger are left untouched).
  * Roster membership: a player with snaps or dropbacks in the window who is NOT
    on the team's reference roster has left the team and now counts as fully
    unavailable. Before, only injury-report status and an explicit prior-week
    RES/CUT/TRD/... row marked anyone out, so departed free agents (the window
    spans the season boundary in weeks 1-3), traded players, and a departed
    starting QB still counted as available.
    Reference roster = PRIOR week's roster; week 1 uses the week-1 roster's
    membership (set at final cutdown, well before kickoff; it can only remove
    players). If more than MEMBERSHIP_MAX_SHARE of a team's window snaps are
    off the reference roster, that roster is treated as an ID/coverage gap and
    membership is skipped for that team-week (audited).
  * 'qb_delta' relabelled: it is the projected QB's efficiency minus the
    team's recent dropback-weighted QB mix, so it is nonzero without any
    change (usually positive: starters beat mop-up/backup snaps).
  * Large-gap warning, footer and manifest no longer claim the availability
    model lacks injury/QB/roster information.

v1.4: adds 'rates_core_avail' = core stats + player availability, using only
information published before kickoff:
  * QB layer: if the usual passer is listed Out/Doubtful (or was moved off the
    roster by the PRIOR week), the next passer by recent dropbacks is projected
    and the feature is the expected net yards per dropback of the projected QB
    minus the team's recent QB mix (each QB shrunk toward backup-level efficiency).
  * Unit availability: for OL, WR/TE, RB, DL, LB and DB, the share of the unit's
    snaps over the team's last 4 games belonging to players listed Out/Doubtful
    (Questionable counts a quarter), or moved to reserve/cut/traded by the prior week.
  Snap counts that reveal who actually played the forecast game are never used.
  Same-week roster STATUS is deliberately ignored because its timing is unknown.

v1.3: adds 'rates_core_adj', the same 8 core stats adjusted for opponents
faced. Each week, using prior games only, a weighted ridge fit splits every
stat into league mean + offense effect + defense effect, so a defense that
faced weak offenses is no longer credited for it.

v1.2: adds a 'rates_core' family (8 stats per side, one per idea) and drops
total yards from box_totals (it equals passing + rushing exactly).

v1.1 changes (design fixes found by code review, applied after v1 results were
seen, so this is a NEW experiment with its own OUTPUT_ROOT and forward ledger):
  * Recipe selection: v1's season-cluster one-SE rule passed all 18 candidates
    with only three validation seasons, so the tie-break (fewest inputs,
    strongest ridge, longest half-life) chose rates_h16_r0.1 regardless of
    data. v1.1 selects the minimum walk-forward log loss; SEs stay diagnostic.
  * Ridge grid extended upward: v1's best candidates sat at the 0.1 boundary.
  * Defensive possession share dropped: it is exactly 100 minus offensive
    share, so the pair was a duplicated feature (mirror coefficients).
  * Backtest starts 2021 with outer tests 2023-2025 (v1 had one outer season).
  * Board redesigned: slate summary, final scores with hit/miss, readable
    feature names, diverging contribution bars, better-side highlighting,
    large model-market gaps flagged, kickoffs in TIMEZONE, scoped CSS.

The user's box-score stat types feed lagged offense and defense profiles.
The logistic composite learns W/L weights from PAST games, not a game's own
final statistics. Team profiles decay by games played; coefficient training
also discounts old games. Half-life, feature family and ridge penalty are
chosen only from earlier-season walk-forward predictions. Coefficients are
refitted each week; the chosen recipe is frozen for the current season.

Market spread is a benchmark only: it never enters the composite fit or
selection. Player availability enters only through the *_avail families;
opponent adjustment only through the *_adj families.

PBP-derived box scores are reconstructions, not guaranteed official totals.
Penalty count is an accepted-yard penalized-play proxy (multiple penalties on
one play may be undercounted); possession time is deduplicated source drive
metadata with coverage checks. Fumbles are assigned to source fumble/recovery
teams where available; fallback attribution is audited. A clean official team-
game CSV can replace PBP aggregation via TEAM_GAME_CSV (schema below).
Historical source revisions and this new design prevent calling backtests
untouched forecasts. Only first snapshots actually written before kickoff go
into the separate forward ledger. Preserve OUTPUT_ROOT across Colab sessions.

References:
https://nflfastr.com/reference/fast_scraper.html
https://nflreadpy.nflverse.com/api/load_functions/
https://www.jmlr.org/papers/v11/cawley10a.html
"""
import os, sys, subprocess, importlib.util, importlib.metadata
for _pkg in ('numpy','pandas','scipy'):
    if importlib.util.find_spec(_pkg) is None:
        subprocess.run([sys.executable,'-m','pip','install','-q',_pkg],check=True)
import datetime as dt
import hashlib
import json
import shutil
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo
from html import escape
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,ndtr
try:
    from IPython.display import display,HTML
    HAS_IPY=True
except ImportError:
    HAS_IPY=False
    def display(x): print(x.to_string(index=False) if isinstance(x,pd.DataFrame) else x)
pd.set_option('display.max_columns',30)
pd.set_option('display.width',240)

# Configuration. Change these before the first run, not in response to one week.
REVISION='boxscore-composite-v1.7'
SEASON=None
CURRENT_WEEK=None
TIMEZONE='America/Los_Angeles'
OUTPUT_NAME=os.environ.get('BOXSCORE_OUTPUT_ROOT','boxscore_cache/nfl_boxscore_output_v1_7')  # new experiment; keep earlier folders untouched
USE_GOOGLE_DRIVE=True        # in Colab: keep outputs, caches, frozen recipe and ledger on Google Drive
DRIVE_MOUNT='/content/drive'
DRIVE_FOLDER='nfl_boxscore'  # folder under MyDrive
OUTPUT_ROOT=Path(OUTPUT_NAME)  # replaced at run time by resolve_output_root()
TEAM_GAME_CSV=None  # Optional authoritative input; schedule still loads from nflverse.
BACKTEST_FIRST_SEASON=2021
WARMUP_SEASONS=2
OUTER_FIRST_SEASON=2023
TEAM_HALF_LIVES=(4.,8.,16.)  # team games: 8 -> a game 8 appearances ago gets half weight
RIDGE_GRID=(.01,.1,1.,10.)   # mean weighted log loss + lambda/2 * ||beta||^2
FEATURE_FAMILIES=('rates_core_avail_cs','rates_core_adj_avail_cs')  # v1.6 winner vs its opponent-adjusted twin
FIT_HALF_LIFE_SEASONS=2.     # separate decay on old labeled training games
OFFSEASON_RETENTION=.5      # each season boundary halves old team-profile weight
PRIOR_EQUIVALENT_GAMES=4.   # shrink sparse histories toward a prior league profile
MAX_HISTORY_SEASONS=3
MIN_TEAM_HISTORY=4
MIN_TRAIN_GAMES=150
SELECTION_RULE='min_log_loss'  # 'one_se' restores v1 behavior
SELECTION_ONE_SE=1.         # used only when SELECTION_RULE=='one_se'; always reported
MARKET_SIGMA=12.37          # fixed spread-to-WP comparator only
FLAG_GAP_PP=10.             # board flags |model - market| at or above this many points
CACHE_MAX_AGE_HOURS=24.
REUSE_CACHE=True
WRITE_FORWARD_LEDGER=True
DIAGNOSTICS='v1.7 opponent-adjusted availability family; market blend, matched-band calibration, pick records, cached availability; Drive storage'
CAL_BAND_EDGES=(0.,.2,.3,.4,.5,.6,.7,.8,1.)  # home-win probability bands shared by model and market
PICK_BAND_EDGES=(.5,.55,.6,.65,.7,.75,.8,1.)  # pick-confidence bands shared by model and market
PICK_BAND_LABELS=('50-55%','55-60%','60-65%','65-70%','70-75%','75-80%','80%+')
SEED=20261005
BOOTSTRAP_REPS=2000
AUDIT=[]

# Required official-CSV counts: one row per game_id/team with opponent, season, week.
# Sacks are TAKEN by the row's offense. Opponent sacks_taken become defensive sacks.
# Net passing includes sack yards; pass_plays includes sacks; pass_attempts excludes them.
# game_seconds = own + opposing possession seconds; NaN is allowed for missing TOP.
COUNTS=('plays','net_yards','net_pass_yards','rush_yards','pass_plays','pass_attempts',
        'rush_attempts','first_downs','third_converted','third_attempts','fumbles_lost',
        'interceptions','sacks_taken','penalties','penalty_yards','top_seconds','game_seconds')
# metric: numerator, denominator (None -> per game), display multiplier.
RATES={
 'plays_per_game':('plays',None,1.),
 'yards_per_play':('net_yards','plays',1.),
 'net_pass_yards_per_pass_play':('net_pass_yards','pass_plays',1.),
 'rush_yards_per_attempt':('rush_yards','rush_attempts',1.),
 'first_down_rate':('first_downs','plays',100.),
 'third_down_pct':('third_converted','third_attempts',100.),
 'fumbles_lost_per_game':('fumbles_lost',None,1.),
 'interception_pct':('interceptions','pass_attempts',100.),
 'sacks_taken_pct':('sacks_taken','pass_plays',100.),
 'penalties_per_game':('penalties',None,1.),
 'penalty_yards_per_game':('penalty_yards',None,1.),
 'possession_share_pct':('top_seconds','game_seconds',100.)}
TOTALS={
 'plays':('plays',None,1.),
 'yards_per_play':('net_yards','plays',1.),'passing_yards_net':('net_pass_yards',None,1.),
 'rushing_yards':('rush_yards',None,1.),'first_downs':('first_downs',None,1.),
 'third_down_pct':('third_converted','third_attempts',100.),
 'fumbles_lost':('fumbles_lost',None,1.),'interceptions':('interceptions',None,1.),
 'sacks_taken':('sacks_taken',None,1.),'penalties':('penalties',None,1.),
 'penalty_yards':('penalty_yards',None,1.),'possession_minutes':('top_seconds',None,1/60.)}
# Core rates: one stat per idea, chosen by definition (not by coefficients).
# Dropped: yards/play (mix of pass and rush efficiency), third-down % (subset of
# first downs), penalty count (same signal as penalty yards), possession share
# (mirrors plays for/against and has missing source data).
RATES_CORE={k:RATES[k] for k in ('plays_per_game','net_pass_yards_per_pass_play','rush_yards_per_attempt',
    'first_down_rate','fumbles_lost_per_game','interception_pct','sacks_taken_pct','penalty_yards_per_game')}
# Relocated franchises map to current codes so one team keeps one history, and
# schedule and play-by-play sources agree (2019 Raiders: OAK vs LV). game_id is untouched.
TEAM_ALIASES={'OAK':'LV','SD':'LAC','STL':'LA'}
TEAM_COLUMNS=('home_team','away_team','team','opponent','posteam','defteam','penalty_team',
              'fumbled_1_team','fumbled_2_team','fumble_recovery_1_team','fumble_recovery_2_team')
def normalize_teams(df):
    for c in TEAM_COLUMNS:
        if c in df: df[c]=df[c].replace(TEAM_ALIASES)
    return df

FAMILIES={'rates_core':RATES_CORE,'rates_core_adj':RATES_CORE,'rates':RATES,'box_totals':TOTALS}
FAMILIES['rates_core_avail']=RATES_CORE
FAMILIES['rates_core_avail_cs']=RATES_CORE
FAMILIES['rates_core_adj_avail_cs']=RATES_CORE
# Opponent-adjusted family -> raw family whose stat definitions it reuses.
ADJUSTED_FAMILIES={'rates_core_adj':'rates_core'}

# Player availability (v1.4+). Availability family -> family whose stats it adds to.
AVAIL_FAMILIES={'rates_core_avail':'rates_core','rates_core_avail_cs':'rates_core',
                'rates_core_adj_avail_cs':'rates_core_adj'}
STATUS_WEIGHT={'Out':1.,'Doubtful':1.,'Questionable':.25}
ROSTER_OUT=('RES','CUT','TRD','RET','EXE')  # read from the PRIOR week's roster only
ROSTER_MEMBERSHIP='most recent roster week before the game (week 1: week-1 roster); off-roster players count as out'
MEMBERSHIP_MAX_SHARE=.5   # above this off-roster snap share, treat the roster as a data gap
AVAIL_WINDOW=4            # team's prior games that define each player's snap share
CS_MIN_GAMES=2            # _cs family: current-season-only window once a team has this many games
QB_RULE='projected QB must be listed at QB on the reference roster; else league backup prior'
NO_QB_LABEL='backup without team dropbacks (league backup prior)'
QB_HALF_LIFE=8.           # team games
QB_PRIOR_DROPBACKS=150.   # shrinkage toward backup-level efficiency
POS_GROUP={'T':'OL','G':'OL','C':'OL','OL':'OL','OT':'OL','OG':'OL','LT':'OL','RT':'OL','LG':'OL','RG':'OL',
 'WR':'WRTE','TE':'WRTE','RB':'RB','FB':'RB','HB':'RB',
 'DE':'DL','DT':'DL','NT':'DL','DL':'DL','EDGE':'DL','LB':'LB','ILB':'LB','OLB':'LB','MLB':'LB',
 'CB':'DB','S':'DB','FS':'DB','SS':'DB','DB':'DB','SAF':'DB'}
OFFENSE_GROUPS=('OL','WRTE','RB'); DEFENSE_GROUPS=('DL','LB','DB')
AVAIL_COLS=('qb_delta',)+tuple(f'{g}_out' for g in OFFENSE_GROUPS+DEFENSE_GROUPS)
AVAIL_COLS_CS=('qb_delta',)+tuple(f'{g}_out_cs' for g in OFFENSE_GROUPS+DEFENSE_GROUPS)
ALL_AVAIL_COLS=tuple(dict.fromkeys(AVAIL_COLS+AVAIL_COLS_CS))
AVAIL_FAMILY_COLS={'rates_core_avail':AVAIL_COLS,'rates_core_avail_cs':AVAIL_COLS_CS,
                   'rates_core_adj_avail_cs':AVAIL_COLS_CS}
AVAIL_LABEL={'qb_delta':'QB: projected starter vs recent QB mix (net yards per dropback)',
 'OL_out':'offensive line snaps unavailable',
 'WRTE_out':'receiver and tight end snaps unavailable','RB_out':'running back snaps unavailable',
 'DL_out':'defensive line snaps unavailable','LB_out':'linebacker snaps unavailable','DB_out':'secondary snaps unavailable'}
AVAIL_LABEL.update({f'{g}_out_cs':AVAIL_LABEL[f'{g}_out']+' (this-season window)' for g in OFFENSE_GROUPS+DEFENSE_GROUPS})
# Allowed possession mirrors own possession (share: exactly 100 - own), so it is
# kept in the displayed profile but not fed to the regression.
REDUNDANT_ALLOWED={'possession_share_pct','possession_minutes'}

# Display vocabulary. "Def" rows describe what opponents did against the team.
METRIC_LABEL={
 'plays_per_game':'plays per game','yards_per_play':'yards per play',
 'net_pass_yards_per_pass_play':'net yards per dropback','rush_yards_per_attempt':'yards per rush',
 'first_down_rate':'first downs per 100 plays','third_down_pct':'third-down conversion %',
 'fumbles_lost_per_game':'fumbles lost per game','interception_pct':'interception %',
 'sacks_taken_pct':'sack %','penalties_per_game':'penalties per game',
 'penalty_yards_per_game':'penalty yards per game','possession_share_pct':'possession %',
 'plays':'plays','yards':'net yards','passing_yards_net':'net passing yards',
 'rushing_yards':'rushing yards','first_downs':'first downs','fumbles_lost':'fumbles lost',
 'interceptions':'interceptions thrown','sacks_taken':'sacks taken','penalties':'penalties',
 'penalty_yards':'penalty yards','possession_minutes':'possession minutes'}
# Direction for the team's own offense; defense ("allowed") rows invert it.
OFFENSE_HIGHER_BETTER={
 'plays_per_game':None,'plays':None,'yards_per_play':True,'net_pass_yards_per_pass_play':True,
 'rush_yards_per_attempt':True,'first_down_rate':True,'third_down_pct':True,
 'fumbles_lost_per_game':False,'interception_pct':False,'sacks_taken_pct':False,
 'penalties_per_game':False,'penalty_yards_per_game':False,'possession_share_pct':True,
 'yards':True,'passing_yards_net':True,'rushing_yards':True,'first_downs':True,
 'fumbles_lost':False,'interceptions':False,'sacks_taken':False,'penalties':False,
 'penalty_yards':False,'possession_minutes':True}


def now_utc(): return dt.datetime.now(dt.timezone.utc)
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def data_hash(df):
    h=hashlib.sha256('|'.join(map(str,df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df,index=False).to_numpy().tobytes())
    return h.hexdigest()
def ll(y,p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); y=np.asarray(y,float)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))
def won(result): return np.nan if pd.isna(result) else 1. if result>0 else 0. if result<0 else .5

def numeric(p,col):
    return pd.to_numeric(p[col],errors='coerce') if col in p else pd.Series(np.nan,index=p.index)

def profile_family(family):
    """Family whose stat profiles a feature family uses (availability families add to a base)."""
    return AVAIL_FAMILIES.get(family,family)

def load_schedule(year):
    import nflreadpy as nfl
    s=nfl.load_schedules(year).to_pandas()
    s=normalize_teams(s[s.game_type=='REG'].copy())
    if s.game_id.duplicated().any(): raise ValueError('Duplicate schedule game IDs')
    s['season']=int(year)
    s['home won']=s.result.map(won)
    s['site']=np.where(s.get('location',pd.Series('',index=s.index)).astype(str).str.lower()=='neutral',0.,1.)
    s['market WP']=ndtr(pd.to_numeric(s.spread_line,errors='coerce')/MARKET_SIGMA)
    AUDIT.append({'source':'schedule','season':int(year),'rows':len(s),'sha256':data_hash(s),'loaded_utc':now_utc().isoformat()})
    return s


def parse_clock(value):
    if pd.isna(value): return np.nan
    bits=str(value).split(':')
    try:
        if len(bits)!=2: return np.nan
        m,s=int(bits[0]),int(bits[1])
        return float(60*m+s) if m>=0 and 0<=s<60 else np.nan
    except (ValueError,TypeError): return np.nan


def possession_by_team(game,home,away):
    """Use the SOURCE drive identifier, not manually renumbered fixed_drive.
    A drive's metadata is counted once. Ambiguous owner/duration -> missing,
    rather than fabricated zeros. TOP is optional for the other stat families.
    """
    if not {'drive','drive_time_of_possession'}<=set(game): return {home:np.nan,away:np.nan}
    totals={home:0.,away:0.}; complete=True
    for _,d in game.dropna(subset=['drive']).groupby('drive',sort=False):
        d=d.sort_values('play_id')
        durations=d.drive_time_of_possession.map(parse_clock).dropna()
        # Determine owner from offensive snaps, excluding the kickoff's kicking posteam.
        snaps=d[(numeric(d,'pass_attempt')==1)|(numeric(d,'rush_attempt')==1)|(numeric(d,'sack')==1)]
        owners=snaps.posteam.dropna().unique()
        if len(owners)!=1:
            if snaps.empty: continue  # no scrimmage snaps; no reliable assigned duration
            complete=False; continue
        owner=owners[0]
        if owner not in totals or durations.empty:
            complete=False; continue
        # Source may fill a drive progressively: its last reported value is the total.
        totals[owner]+=float(durations.iloc[-1])
    total=sum(totals.values())
    if not complete or min(totals.values())<=0 or not 3300<=total<=5400:
        return {home:np.nan,away:np.nan}
    return totals


def aggregate_boxscores(p,schedule,year):
    req={'game_id','play_id','season_type','posteam','play_type','yards_gained',
         'pass_attempt','rush_attempt','sack','interception','fumble_lost',
         'third_down_converted','third_down_failed','first_down_rush','first_down_pass','first_down_penalty',
         'penalty','penalty_team','penalty_yards','two_point_attempt'}
    missing=req-set(p)
    if missing: raise ValueError(f'PBP {year}: missing required fields {sorted(missing)}')
    if p.duplicated(['game_id','play_id']).any(): raise ValueError('Duplicate PBP play keys')
    # Full games, all score states. Do NOT reuse the v4 win-probability-filtered PBP.
    p=p[(p.season_type=='REG')&(numeric(p,'two_point_attempt')!=1)].copy()
    if 'play_deleted' in p: p=p[numeric(p,'play_deleted')!=1]
    finals=schedule[schedule.result.notna()]
    grouped={gid:g for gid,g in p.groupby('game_id',sort=False)}
    rows=[]; missing_games=[]; fumble_fallbacks=0; ambiguous_fumbles=0
    for g in finals.itertuples(index=False):
        if g.game_id not in grouped:
            missing_games.append(g.game_id); continue
        d=grouped[g.game_id].copy()
        teams_in_pbp=set(d.posteam.dropna())
        if not {g.home_team,g.away_team}<=teams_in_pbp:
            raise ValueError(f'{g.game_id}: schedule teams {g.home_team}/{g.away_team} vs play-by-play teams '
                             f'{sorted(teams_in_pbp)}. Add the mismatched code to TEAM_ALIASES.')
        valid=d.play_type.ne('no_play')
        passplay=((numeric(d,'pass_attempt')==1)|(numeric(d,'sack')==1))&valid
        runplay=(numeric(d,'rush_attempt')==1)&valid&~passplay
        snap=passplay|runplay
        if not snap.any(): missing_games.append(g.game_id); continue
        top=possession_by_team(d,g.home_team,g.away_team)
        fumbles={g.home_team:0.,g.away_team:0.}
        for _,r in d[(numeric(d,'fumble_lost')==1)&valid].iterrows():
            attributed=[]
            for n in (1,2):
                ft,rt=r.get(f'fumbled_{n}_team'),r.get(f'fumble_recovery_{n}_team')
                if pd.notna(ft) and pd.notna(rt) and ft!=rt and ft in fumbles: attributed.append(ft)
            if attributed:
                for team in attributed: fumbles[team]+=1
            else:
                ft=r.get('fumbled_1_team')
                team=ft if pd.notna(ft) and ft in fumbles else r.posteam
                if team in fumbles: fumbles[team]+=1; fumble_fallbacks+=1
                else: ambiguous_fumbles+=1
        for team,opp in ((g.home_team,g.away_team),(g.away_team,g.home_team)):
            own=d.posteam.eq(team)
            ps,rs=passplay&own,runplay&own
            nplays=int((snap&own).sum())
            if nplays==0: raise ValueError(f'{g.game_id}: no offensive plays for {team}')
            yard=numeric(d,'yards_gained')
            fd=pd.concat([numeric(d,c) for c in ('first_down_rush','first_down_pass','first_down_penalty')],axis=1).max(axis=1)
            # Count penalized plays with nonzero assessed yards. Declined/offset penalties
            # are excluded by their zero/missing yard field; multipenalty ambiguity remains.
            penalty=(numeric(d,'penalty')==1)&d.penalty_team.eq(team)&numeric(d,'penalty_yards').abs().gt(0)
            r={'game_id':g.game_id,'season':int(year),'week':int(g.week),'team':team,'opponent':opp,
               'plays':float(nplays),'net_pass_yards':float(yard[ps].sum(min_count=1)),
               'rush_yards':float(yard[rs].sum(min_count=1)),
               'pass_plays':float(ps.sum()),'pass_attempts':float((ps&(numeric(d,'sack')!=1)).sum()),
               'rush_attempts':float(rs.sum()),'first_downs':float(fd[own].sum(min_count=1)),
               'third_converted':float(numeric(d,'third_down_converted')[own].sum(min_count=1)),
               'third_attempts':float((numeric(d,'third_down_converted')[own].fillna(0)+numeric(d,'third_down_failed')[own].fillna(0)).sum()),
               'fumbles_lost':fumbles[team],'interceptions':float(((numeric(d,'interception')==1)&ps).sum()),
               'sacks_taken':float(((numeric(d,'sack')==1)&ps).sum()),
               'penalties':float(penalty.sum()),'penalty_yards':float(numeric(d,'penalty_yards')[penalty].abs().sum()),
               'top_seconds':top[team],'game_seconds':top[team]+top[opp]}
            # Zero attempts is valid: no rushing/passing yardage, not a missing total.
            if not ps.any(): r['net_pass_yards']=0.
            if not rs.any(): r['rush_yards']=0.
            r['net_yards']=r['net_pass_yards']+r['rush_yards']
            rows.append(r)
    box=pd.DataFrame(rows)
    AUDIT.append({'source':'pbp_box_reconstruction','season':int(year),'team_games':len(box),
        'missing_games':missing_games,'fumble_fallback_plays':fumble_fallbacks,'ambiguous_fumble_plays':ambiguous_fumbles,
        'top_missing_team_games':int(box.top_seconds.isna().sum()) if len(box) else 0,
        'penalty_definition':'nonzero assessed-yard penalized plays; not guaranteed official multiple-penalty counts'})
    if missing_games: warnings.warn(f'{year}: {len(missing_games)} final games lack PBP; missing histories are audited.')
    return box


def validate_boxes(box):
    required={'game_id','season','week','team','opponent',*COUNTS}
    if not required<=set(box): raise ValueError(f'Team-game data missing: {sorted(required-set(box))}')
    if box.duplicated(['game_id','team']).any(): raise ValueError('Duplicate team-game rows')
    for _,g in box.groupby('game_id'):
        if len(g)!=2 or set(g.team)!=set(g.opponent): raise ValueError('Each box-score game needs two reciprocal team rows')
    if (box.plays<=0).any(): raise ValueError('Nonpositive play count')
    if not np.allclose(box.net_yards,box.net_pass_yards+box.rush_yards,equal_nan=True):
        raise ValueError('net_yards must equal net_pass_yards + rush_yards')
    return box.sort_values(['season','week','game_id','team']).reset_index(drop=True)


def aggregate_qb(p,schedule):
    """Passer-level dropbacks and net yards per team-game (sacks included)."""
    p=p[(p.season_type=='REG')&(numeric(p,'two_point_attempt')!=1)]
    if 'play_deleted' in p: p=p[numeric(p,'play_deleted')!=1]
    if 'passer_player_id' not in p: return pd.DataFrame(columns=['game_id','season','week','team','gsis_id','name','dropbacks','net_yards','leader'])
    db=p[p.play_type.ne('no_play')&((numeric(p,'pass_attempt')==1)|(numeric(p,'sack')==1))&p.passer_player_id.notna()].copy()
    db['ny']=numeric(db,'yards_gained').fillna(0.)
    q=db.groupby(['game_id','posteam','passer_player_id'],as_index=False).agg(
        dropbacks=('play_id','size'),net_yards=('ny','sum'),name=('passer_player_name','first'))
    q=q.rename(columns={'posteam':'team','passer_player_id':'gsis_id'})
    q=q.merge(schedule.loc[schedule.result.notna(),['game_id','season','week']],on='game_id',how='inner')
    q['leader']=q.dropbacks.eq(q.groupby(['game_id','team']).dropbacks.transform('max'))
    return q


def load_boxes(year,schedule):
    path=OUTPUT_ROOT/'cache'/f'boxscores_{year}_{REVISION}.csv'; meta=path.with_suffix('.json')
    qbpath=OUTPUT_ROOT/'cache'/f'qb_{year}_{REVISION}.csv'
    schedule_hash=data_hash(schedule[['game_id','week','result','home_team','away_team']])
    if REUSE_CACHE and path.exists() and meta.exists():
        m=json.loads(meta.read_text())
        age=(now_utc()-dt.datetime.fromisoformat(m['generated_utc'])).total_seconds()/3600
        if (0<=age<CACHE_MAX_AGE_HOURS and m['schedule_hash']==schedule_hash and qbpath.exists()
                and m['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()):
            AUDIT.append({'cache':str(path),**m}); return pd.read_csv(path),pd.read_csv(qbpath)
    if schedule.result.notna().sum()==0: return pd.DataFrame(columns=['game_id','season','week','team','opponent',*COUNTS]),None
    import nflreadpy as nfl
    raw=nfl.load_pbp(year)
    cols=['game_id','play_id','season_type','posteam','play_type','yards_gained','pass_attempt','rush_attempt',
          'sack','interception','fumble_lost','third_down_converted','third_down_failed','first_down_rush',
          'first_down_pass','first_down_penalty','penalty','penalty_team','penalty_yards','two_point_attempt',
          'drive','drive_time_of_possession','play_deleted','fumbled_1_team','fumbled_2_team',
          'fumble_recovery_1_team','fumble_recovery_2_team','passer_player_id','passer_player_name']
    p=raw.select([c for c in cols if c in raw.columns]).to_pandas(); del raw
    source_hash=data_hash(p); p=normalize_teams(p)
    box=aggregate_boxscores(p,schedule,year); qb=aggregate_qb(p,schedule); del p
    if box.empty: return box,qb
    box=validate_boxes(box)
    path.parent.mkdir(parents=True,exist_ok=True); box.to_csv(path,index=False); qb.to_csv(qbpath,index=False)
    m={'generated_utc':now_utc().isoformat(),'schedule_hash':schedule_hash,'pbp_sha256':source_hash,
       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'aggregation_audit':AUDIT[-1]}
    dump(meta,m); return box,qb


def _nflverse_parquet(tag,year):
    return pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/{tag}/{tag}_{year}.parquet')


def load_availability(years,season):
    """Weekly injury reports, weekly rosters and snap counts (REG only), with
    snap counts mapped to GSIS ids. Completed seasons are cached; the current
    season reloads every run."""
    import nflreadpy as nfl
    cache=OUTPUT_ROOT/'cache'; cache.mkdir(parents=True,exist_ok=True)
    players=None; out={'inj':[],'rost':[],'snaps':[]}
    for y in years:
        paths={k:cache/f'avail_{k}_{y}.csv' for k in out}
        if y<season and all(p.exists() for p in paths.values()):
            for k,p in paths.items(): out[k].append(pd.read_csv(p))
            continue
        try: inj=nfl.load_injuries(y).to_pandas()
        except Exception: inj=_nflverse_parquet('injuries',y)  # nflreadpy can fail on newer files
        inj=normalize_teams(inj[inj.game_type=='REG'].copy())
        inj['weight']=inj.report_status.map(STATUS_WEIGHT).fillna(0.)
        inj=inj.dropna(subset=['gsis_id']).groupby(['season','week','team','gsis_id'],as_index=False).weight.max()
        rost=nfl.load_rosters_weekly(y).to_pandas()
        if 'game_type' in rost: rost=rost[rost.game_type=='REG']
        rost=normalize_teams(rost.dropna(subset=['gsis_id'])[['season','week','team','gsis_id','position','status']].copy())
        snaps=nfl.load_snap_counts(y).to_pandas(); snaps=normalize_teams(snaps[snaps.game_type=='REG'].copy())
        if players is None:
            pl=nfl.load_players().to_pandas().dropna(subset=['pfr_id','gsis_id'])
            players=dict(zip(pl.pfr_id,pl.gsis_id))
        snaps['gsis_id']=snaps.pfr_player_id.map(players)
        tot=(snaps.offense_snaps+snaps.defense_snaps).sum()
        mapped=(snaps.offense_snaps+snaps.defense_snaps)[snaps.gsis_id.notna()].sum()
        AUDIT.append({'source':'availability','season':int(y),'injury_rows':len(inj),'roster_rows':len(rost),
                      'snap_rows':len(snaps),'snap_share_mapped_to_gsis':float(mapped/tot) if tot else None})
        snaps=snaps.dropna(subset=['gsis_id'])[['season','week','game_id','team','gsis_id','position','offense_snaps','defense_snaps']]
        for k,df in (('inj',inj),('rost',rost),('snaps',snaps)):
            if y<season: df.to_csv(paths[k],index=False)
            out[k].append(df)
    return {k:pd.concat(v,ignore_index=True) for k,v in out.items()}


def availability_table(targets,qb,snaps,inj,rost,audit=None):
    """One row per (season, week, team) to forecast. Uses the week's injury
    report, the team's most recent roster before the game (status and
    membership; week 1 uses the week-1 roster), and snaps/QB play from earlier
    games. A player in the window who is not on the reference roster has left
    and counts as out. '_cs' unit columns use current-season games only once
    the team has CS_MIN_GAMES of them."""
    inj_map={k:dict(zip(g.gsis_id,g.weight)) for k,g in inj.groupby(['season','week','team'])}
    rost_out={k:set(g.gsis_id[g.status.isin(ROSTER_OUT)]) for k,g in rost.groupby(['season','week','team'])}
    members={k:set(g.gsis_id[~g.status.isin(ROSTER_OUT)]) for k,g in rost.groupby(['season','week','team'])}
    qb_ids={k:set(g.gsis_id[(g.position=='QB')&~g.status.isin(ROSTER_OUT)]) for k,g in rost.groupby(['season','week','team'])}
    roster_weeks={}
    for (rs,rw,rt) in members: roster_weeks.setdefault((int(rs),rt),[]).append(int(rw))
    def ref_week_for(y,w,t):
        # Teams on bye have no roster row that week, so use the latest week before the game.
        if w==1: return 1 if (y,1,t) in members else None
        prev=[wk for wk in roster_weeks.get((y,t),()) if wk<w]
        return max(prev) if prev else None
    snaps=snaps.copy(); snaps['group']=snaps.position.map(POS_GROUP)
    snaps=snaps[snaps.group.notna()].copy()
    snaps['snaps']=np.where(snaps.group.isin(OFFENSE_GROUPS),snaps.offense_snaps,snaps.defense_snaps)
    by_team={t:g for t,g in snaps.groupby('team')}
    qb_team={t:g for t,g in qb.groupby('team')} if qb is not None and len(qb) else {}
    prior_cache={}
    def backup_prior(y,w):
        if (y,w) not in prior_cache:
            v=np.nan
            if qb is not None and len(qb):
                h=qb[before(qb,y,w)&(qb.season>=y-MAX_HISTORY_SEASONS)&~qb.leader.astype(bool)]
                if h.dropbacks.sum()>0: v=float(h.net_yards.sum()/h.dropbacks.sum())
            prior_cache[(y,w)]=v
        return prior_cache[(y,w)]
    stats={'team_weeks':0,'no_reference_roster':0,'reference_older_than_prior_week':0,
           'membership_skipped_as_data_gap':0,'qb_projected_from_backup_prior':0,
           'window_snaps':0.,'off_roster_snaps':0.}
    rows=[]
    for r in targets[['season','week','team']].drop_duplicates().itertuples(index=False):
        y,w,t=int(r.season),int(r.week),r.team
        stats['team_weeks']+=1
        ref_week=ref_week_for(y,w,t)
        out=dict(inj_map.get((y,w,t),{}))
        if ref_week is not None and w>1:
            for pid in rost_out.get((y,ref_week,t),()): out[pid]=1.
        ref=members.get((y,ref_week,t)) if ref_week is not None else None
        if ref is not None and w>1 and ref_week<w-1: stats['reference_older_than_prior_week']+=1
        rec={'season':y,'week':w,'team':t,'qb_expected':None,'qb_usual':None,
             'roster_reference_week':ref_week if ref is not None else np.nan,'off_roster_snap_share':np.nan}
        g=by_team.get(t); h=None; hcs=None
        if g is not None:
            past=g[before(g,y,w)]
            games=past[['season','week','game_id']].drop_duplicates().sort_values(['season','week'])
            h=past[past.game_id.isin(games.tail(AVAIL_WINDOW).game_id)]
            cur=games[games.season==y]
            hcs=past[past.game_id.isin(cur.tail(AVAIL_WINDOW).game_id)] if len(cur)>=CS_MIN_GAMES else h
        if ref is None:
            stats['no_reference_roster']+=1
        elif h is not None and h.snaps.sum()>0:
            per_all=h.groupby('gsis_id').snaps.sum()
            share=float(per_all[~per_all.index.isin(ref)].sum()/per_all.sum())
            rec['off_roster_snap_share']=share
            if share>MEMBERSHIP_MAX_SHARE:
                ref=None; stats['membership_skipped_as_data_gap']+=1  # likely an ID/coverage gap, not departures
            else:
                stats['window_snaps']+=float(per_all.sum()); stats['off_roster_snaps']+=share*float(per_all.sum())
        def unavail(pid,ref=ref,out=out):
            return 1. if (ref is not None and pid not in ref) else out.get(pid,0.)
        for grp in OFFENSE_GROUPS+DEFENSE_GROUPS: rec[f'{grp}_out']=np.nan; rec[f'{grp}_out_cs']=np.nan
        for suffix,frame in (('',h),('_cs',hcs)):
            if frame is None: continue
            for grp in OFFENSE_GROUPS+DEFENSE_GROUPS:
                per=frame[frame.group==grp].groupby('gsis_id').snaps.sum(); tot=per.sum()
                if tot>0: rec[f'{grp}_out{suffix}']=float(sum(v*unavail(pid) for pid,v in per.items())/tot)
        q=qb_team.get(t); rec['qb_delta']=np.nan
        if q is not None:
            hq=q[before(q,y,w)&(q.season>=y-1)]
            prior=backup_prior(y,w)
            if len(hq) and np.isfinite(prior):
                order=hq[['season','week','game_id']].drop_duplicates().sort_values(['season','week']).reset_index(drop=True)
                age=pd.Series(np.arange(len(order)-1,-1,-1,dtype=float),index=order.game_id)
                wt=np.exp2(-hq.game_id.map(age).to_numpy()/QB_HALF_LIFE)*np.power(OFFSEASON_RETENTION,y-hq.season.to_numpy(float))
                agg=pd.DataFrame({'gsis_id':hq.gsis_id.to_numpy(),'name':hq['name'].to_numpy(),
                                  'n':wt*hq.dropbacks.to_numpy(),'yds':wt*hq.net_yards.to_numpy()}).groupby('gsis_id').agg(
                    n=('n','sum'),yds=('yds','sum'),name=('name','last'))
                agg['eff']=(agg.yds+QB_PRIOR_DROPBACKS*prior)/(agg.n+QB_PRIOR_DROPBACKS)
                # The mix keeps departed passers: it describes what the team's profile stats reflect.
                mix=float((agg.n*agg.eff).sum()/agg.n.sum())
                ok=agg[[unavail(pid)<1. for pid in agg.index]]
                # Only rostered quarterbacks can be projected (no gadget passers).
                if ref is not None: ok=ok[ok.index.isin(qb_ids.get((y,ref_week,t),set()))]
                proj=float(ok.loc[ok.n.idxmax(),'eff']) if len(ok) else prior
                if not len(ok): stats['qb_projected_from_backup_prior']+=1
                rec['qb_delta']=proj-mix
                rec['qb_usual']=agg.loc[agg.n.idxmax(),'name']
                rec['qb_expected']=ok.loc[ok.n.idxmax(),'name'] if len(ok) else NO_QB_LABEL
        rows.append(rec)
    if audit is not None:
        audit.append({'source':'availability_roster_membership','rule':ROSTER_MEMBERSHIP,'qb_rule':QB_RULE,**stats,
            'off_roster_snap_share':stats['off_roster_snaps']/stats['window_snaps'] if stats['window_snaps'] else None})
    return pd.DataFrame(rows)

AVAIL_STAT_KEYS=('team_weeks','no_reference_roster','reference_older_than_prior_week','membership_skipped_as_data_gap',
                 'qb_projected_from_backup_prior','window_snaps','off_roster_snaps')


def combine_avail_stats(stats):
    tot={k:sum(float(st.get(k,0)) for st in stats) for k in AVAIL_STAT_KEYS}
    for k in AVAIL_STAT_KEYS:
        if k not in ('window_snaps','off_roster_snaps'): tot[k]=int(tot[k])
    tot['off_roster_snap_share']=tot['off_roster_snaps']/tot['window_snaps'] if tot['window_snaps'] else None
    return {'source':'availability_roster_membership','rule':ROSTER_MEMBERSHIP,'qb_rule':QB_RULE,**tot}


def availability_cached(targets,qb,snaps,inj,rost,season,audit=None):
    """availability_table for every season, reading completed seasons from a
    per-season cache keyed on the config signature and all inputs the season
    can touch. The current season is always rebuilt. Returns (table, info)."""
    sig,_=config_signature()
    cache=OUTPUT_ROOT/'cache'; cache.mkdir(parents=True,exist_ok=True)
    def window(df,y):
        if df is None or not len(df): return pd.DataFrame({'empty':[]})
        return df[(df.season>=y-MAX_HISTORY_SEASONS)&(df.season<=y)].sort_values(list(df.columns)[:4]).reset_index(drop=True)
    parts=[]; stats=[]; info={'cached_seasons':[],'rebuilt_seasons':[]}
    for y in sorted(int(v) for v in targets.season.unique()):
        t=targets[targets.season==y][['season','week','team']].drop_duplicates().sort_values(['week','team']).reset_index(drop=True)
        if y>=season:
            st=[]; parts.append(availability_table(t,qb,snaps,inj,rost,audit=st)); stats+=st
            info['rebuilt_seasons'].append(y); continue
        h=hashlib.sha256((REVISION+sig).encode()); h.update(data_hash(t).encode())
        for df in (qb,snaps,inj,rost): h.update(data_hash(window(df,y)).encode())
        path=cache/f'avail_features_{y}_{h.hexdigest()[:16]}.pkl'; meta=path.with_suffix('.json')
        df=None
        if REUSE_CACHE and path.exists() and meta.exists():
            try:
                df=pd.read_pickle(path); st=[json.loads(meta.read_text())['stats']]
            except Exception: df=None  # unreadable (e.g. pandas version change): rebuild
        if df is None:
            st=[]; df=availability_table(t,qb,snaps,inj,rost,audit=st)
            for old in cache.glob(f'avail_features_{y}_*'):
                if old.stem!=path.stem: old.unlink()
            df.to_pickle(path); dump(meta,{'revision':REVISION,'season':y,'stats':st[0],'created_utc':now_utc().isoformat()})
            info['rebuilt_seasons'].append(y)
        else: info['cached_seasons'].append(y)
        parts.append(df); stats+=st
    if audit is not None: audit.append({**combine_avail_stats(stats),**info})
    out=pd.concat(parts,ignore_index=True)
    for c in ('qb_expected','qb_usual'):  # same dtype and missing marker however seasons were combined
        out[c]=out[c].astype(object).where(out[c].notna(),None)
    return out,info


# Lagged profiles. No final-game stat can enter its own feature row.
def before(df,year,week):
    return (df.season<year)|((df.season==year)&(df.week<week))


def metric_profile(values,weights,league,specs):
    """Ratios of decayed counts, with four league-average pseudo-games.
    Rates are NOT means of per-game percentages; valid numerator/denominator
    observations share the same weights. Missing time does not become zero TOP.
    """
    ix={c:i for i,c in enumerate(COUNTS)}
    out={}
    for name,(num,den,mult) in specs.items():
        a=values[:,ix[num]]
        b=np.ones(len(values)) if den is None else values[:,ix[den]]
        la=league[:,ix[num]]
        lb=np.ones(len(league)) if den is None else league[:,ix[den]]
        lok=np.isfinite(la)&np.isfinite(lb)&(lb>0)
        if not lok.any(): out[name]=np.nan; continue
        pa,pb=float(la[lok].mean()),float(lb[lok].mean())
        ok=np.isfinite(a)&np.isfinite(b)&(b>0)
        numer=float(np.dot(weights[ok],a[ok]))+PRIOR_EQUIVALENT_GAMES*pa
        denom=float(np.dot(weights[ok],b[ok]))+PRIOR_EQUIVALENT_GAMES*pb
        out[name]=mult*numer/denom if denom>0 else np.nan
    return out


def opponent_adjust(offense,defense,num,den,weights,prior_games):
    """Weighted ridge fit of value ~ mu + off[offense team] + def[defense team].
    Observation weight = time decay x denominator, so the fit behaves like a
    ratio of sums. Team effects shrink toward zero with the strength of
    prior_games league-average games, matching the raw profiles' smoothing.
    Returns mu and dicts of offense and defense effects (0 = league average).
    """
    offense=np.asarray(offense); defense=np.asarray(defense)
    num=np.asarray(num,float); den=np.asarray(den,float); weights=np.asarray(weights,float)
    ok=np.isfinite(num)&np.isfinite(den)&(den>0)&np.isfinite(weights)&(weights>0)
    if not ok.any(): return np.nan,{},{}
    teams=sorted(set(offense[ok])|set(defense[ok])); ix={t:i for i,t in enumerate(teams)}; T=len(teams)
    n=int(ok.sum()); X=np.zeros((n,1+2*T)); X[:,0]=1.
    X[np.arange(n),1+np.array([ix[t] for t in offense[ok]])]=1.
    X[np.arange(n),1+T+np.array([ix[t] for t in defense[ok]])]=1.
    y=num[ok]/den[ok]; w=weights[ok]*den[ok]
    A=X.T@(w[:,None]*X); b=X.T@(w*y)
    kappa=prior_games*float(np.mean(den[ok]))
    A[1:,1:]+=kappa*np.eye(2*T)
    beta=np.linalg.solve(A,b)
    return float(beta[0]),dict(zip(teams,beta[1:1+T])),dict(zip(teams,beta[1+T:]))


def lagged_features(box,schedules,half_life,avail=None):
    box=validate_boxes(box)
    av=avail.set_index(['season','week','team']) if avail is not None and len(avail) else None
    opp=box[['game_id','team',*COUNTS]].rename(columns={'team':'opponent',**{c:'opp_'+c for c in COUNTS}})
    both=box.merge(opp,on=['game_id','opponent'],how='left',validate='one_to_one')
    records=[]
    for (year,week),games in schedules.groupby(['season','week'],sort=True):
        year,week=int(year),int(week)
        hist=both[before(both,year,week)&(both.season>=year-MAX_HISTORY_SEASONS)]
        league=hist[list(COUNTS)].to_numpy(float)
        profiles={}; counts={}
        for team in set(games.home_team)|set(games.away_team):
            h=hist[hist.team==team].sort_values(['season','week','game_id'])
            n=len(h); counts[team]=n
            age=np.arange(n-1,-1,-1,dtype=float)
            weights=np.exp2(-age/half_life)*np.power(OFFSEASON_RETENTION,year-h.season.to_numpy(float))
            profiles[team]={}
            for role,cols in [('for',list(COUNTS)),('allowed',['opp_'+c for c in COUNTS])]:
                vals=h[cols].to_numpy(float)
                for family,specs in FAMILIES.items():
                    if family in ADJUSTED_FAMILIES or family in AVAIL_FAMILIES: continue
                    for metric,value in metric_profile(vals,weights,league,specs).items():
                        profiles[team][f'{role}__{family}__{metric}']=value
        # Opponent-adjusted ratings from the same prior games. Age is counted in
        # league week-slots (about one game per team), with the same offseason discount.
        ix={c:i for i,c in enumerate(COUNTS)}
        if len(hist):
            slot=(hist.season.astype(int)*100+hist.week.astype(int)).to_numpy()
            uniq=np.unique(slot); slot_age=(len(uniq)-1-np.searchsorted(uniq,slot)).astype(float)
            decay=np.exp2(-slot_age/half_life)*np.power(OFFSEASON_RETENTION,year-hist.season.to_numpy(float))
            hv=hist[list(COUNTS)].to_numpy(float)
        for adj_family,base in ADJUSTED_FAMILIES.items():
            for metric,(num,den,mult) in FAMILIES[base].items():
                if len(hist):
                    a=hv[:,ix[num]]; d=np.ones(len(hv)) if den is None else hv[:,ix[den]]
                    mu,off,dfn=opponent_adjust(hist.team.to_numpy(),hist.opponent.to_numpy(),a,d,decay,PRIOR_EQUIVALENT_GAMES)
                else: mu,off,dfn=np.nan,{},{}
                for team in profiles:
                    profiles[team][f'for__{adj_family}__{metric}']=mult*(mu+off.get(team,0.))
                    profiles[team][f'allowed__{adj_family}__{metric}']=mult*(mu+dfn.get(team,0.))
        for _,g in games.iterrows():
            r={'game_id':g.game_id,'season':year,'week':week,'home':g.home_team,'away':g.away_team,
               'gameday':str(g.gameday),'gametime':g.gametime,'site':g.site,'result':g.result,
               'home_score':g.get('home_score',np.nan),'away_score':g.get('away_score',np.nan),
               'home won':g['home won'],'spread_line':g.spread_line,'market_wp':g['market WP'],
               'home_history_games':counts[g.home_team],'away_history_games':counts[g.away_team],
               'ready':min(counts[g.home_team],counts[g.away_team])>=MIN_TEAM_HISTORY,
               'train_through_week':week-1,'profile_half_life':half_life}
            for name,h in profiles[g.home_team].items():
                a=profiles[g.away_team][name]
                r['home__'+name],r['away__'+name],r['d__'+name]=h,a,h-a
            for side,team in (('home',g.home_team),('away',g.away_team)):
                key=(year,week,team)
                row=av.loc[key] if av is not None and key in av.index else None
                for c in ALL_AVAIL_COLS: r[f'{side}__avail__{c}']=float(row[c]) if row is not None else np.nan
                for c in ('qb_expected','qb_usual'): r[f'{side}_{c}']=row[c] if row is not None else None
            for c in ALL_AVAIL_COLS: r['d__avail__'+c]=r['home__avail__'+c]-r['away__avail__'+c]
            records.append(r)
    return pd.DataFrame(records).sort_values(['season','week','game_id']).reset_index(drop=True)


def feature_names(family):
    if family in AVAIL_FAMILIES:
        return feature_names(AVAIL_FAMILIES[family])+[f'd__avail__{c}' for c in AVAIL_FAMILY_COLS[family]]
    return ['site']+[f'd__{role}__{family}__{metric}' for role in ('for','allowed') for metric in FAMILIES[family]
                     if not (role=='allowed' and metric in REDUNDANT_ALLOWED)]


def checked_optimize(obj,p):
    opt=minimize(obj,np.zeros(p),jac=True,method='L-BFGS-B',options={'maxiter':1500,'gtol':1e-8,'ftol':1e-12})
    if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all() or np.max(np.abs(obj(opt.x)[1]))>2e-5:
        raise RuntimeError(f'Composite fit failed: {opt.message}')
    return opt.x


def fit_composite(train,recipe,year,week,baseline=False):
    t=train[train['home won'].isin([0.,1.])&train.ready].copy()
    if len(t)<MIN_TRAIN_GAMES: raise ValueError(f'Only {len(t)} eligible labeled games; need {MIN_TRAIN_GAMES}')
    if not before(t,year,week).all(): raise ValueError('Training includes current/future week')
    names=['site'] if baseline else feature_names(recipe['family'])
    raw=t[names].to_numpy(float)
    missing=~np.isfinite(raw)
    # Zero difference represents an unavailable/league-neutral matchup, not a future mean.
    X=np.where(missing,0.,raw)
    age=((year-t.season.to_numpy())*19+(week-t.week.to_numpy()))/(19*FIT_HALF_LIFE_SEASONS)
    weights=np.exp2(-age); weights/=weights.sum()
    mu=np.sum(weights[:,None]*X,axis=0)
    scale=np.sqrt(np.sum(weights[:,None]*(X-mu)**2,axis=0))
    scale=np.where(scale>1e-8,scale,1.); scale[0]=1.
    Z=X/scale  # scale without centering: neutral-site team symmetry stays exact
    y=t['home won'].to_numpy(float)
    lam=float(recipe['ridge']) if not baseline else .01
    penalty=np.full(len(names),lam); penalty[0]*=.1
    def obj(beta):
        eta=Z@beta
        loss=np.sum(weights*(np.logaddexp(0,eta)-y*eta))+.5*np.sum(penalty*beta*beta)
        grad=Z.T@(weights*(expit(eta)-y))+penalty*beta
        return loss,grad
    beta=checked_optimize(obj,len(names))
    return {'names':names,'scale':scale.tolist(),'beta':beta.tolist(),'recipe':recipe,
            'fit_season':int(year),'fit_week':int(week),'training_games':len(t),
            'training_max_season':int(t.season.max()),
            'training_max_week_in_latest_season':int(t.loc[t.season==t.season.max(),'week'].max()),
            'effective_training_games':float(1/np.sum(weights*weights)),
            'training_missing_fractions':dict(zip(names,missing.mean(axis=0).tolist()))}


def apply_fit(df,fit,contributions=False):
    x=df[fit['names']].to_numpy(float)
    x=np.where(np.isfinite(x),x,0.)/np.asarray(fit['scale'])
    parts=x*np.asarray(fit['beta'])
    score=parts.sum(axis=1)
    return (expit(score),score,parts) if contributions else expit(score)


def recipe_key(r): return f'{r["family"]}_h{r["half_life"]:g}_r{r["ridge"]:g}'
def candidates():
    return [{'family':f,'half_life':h,'ridge':r} for f in FEATURE_FAMILIES for h in TEAM_HALF_LIVES for r in RIDGE_GRID]


def walk_forward_grid(features,last_historical_season):
    """Fit each candidate separately before EVERY evaluation week.
    Hyperparameter selection is performed downstream using older seasons only.
    Coefficients may adapt to earlier weeks within the held-out year: operational
    walk-forward evaluation, not an annual fixed-coefficient test.
    """
    first=features[TEAM_HALF_LIVES[0]]
    valid=(first.season>=BACKTEST_FIRST_SEASON)&(first.season<=last_historical_season)&first['home won'].isin([0.,1.])&first.ready
    out=first.loc[valid,['game_id','season','week','home','away','home won','market_wp','result','spread_line']].copy()
    for recipe in candidates():
        key=recipe_key(recipe); print(f'  Walk-forward {key}')
        f=features[recipe['half_life']]
        pred=pd.Series(np.nan,index=f.index)
        for (year,week),test in f.loc[valid].groupby(['season','week'],sort=True):
            tr=f[before(f,int(year),int(week))]
            fit=fit_composite(tr,recipe,int(year),int(week))
            pred.loc[test.index]=apply_fit(test,fit)
        out['p__'+key]=pred.loc[valid].to_numpy()
    out['homefield_wp']=np.nan
    for (year,week),test in first.loc[valid].groupby(['season','week'],sort=True):
        fit=fit_composite(first[before(first,int(year),int(week))],candidates()[0],int(year),int(week),baseline=True)
        out.loc[test.index,'homefield_wp']=apply_fit(test,fit)
    if out.filter(regex='^p__').isna().any().any(): raise ValueError('Incomplete walk-forward candidate predictions')
    return out.reset_index(drop=True)


def paired_season_se(delta,seasons):
    d=np.asarray(delta,float); n=len(d)
    sums=pd.Series(d-d.mean()).groupby(np.asarray(seasons)).sum()
    g=len(sums)
    if g<2: return np.nan
    return float(np.sqrt(g/(g-1)*np.square(sums).sum()/n**2))


def select_recipe(oof):
    if oof.season.nunique()<2: raise ValueError('Need at least two completed validation seasons for recipe selection')
    y=oof['home won'].to_numpy(float)
    errors={recipe_key(r):ll(y,oof['p__'+recipe_key(r)]) for r in candidates()}
    best=min(errors,key=lambda k:errors[k].mean())
    rows=[]; eligible=[]
    for r in candidates():
        key=recipe_key(r); delta=errors[key]-errors[best]
        se=paired_season_se(delta,oof.season)
        ok=float(delta.mean())<=SELECTION_ONE_SE*se+1e-12
        if ok: eligible.append(r)
        rows.append({**r,'key':key,'log loss':float(errors[key].mean()),
            'LL worse than minimum':float(delta.mean()),'paired season SE':se,'within simplicity tolerance':bool(ok),
            'games':len(oof),'validation seasons':int(oof.season.nunique())})
    if SELECTION_RULE=='one_se':
        # v1 rule: fewest inputs, strongest shrinkage, then longer memory among similar losses.
        choice=min(eligible,key=lambda r:(len(feature_names(r['family'])),-r['ridge'],-r['half_life']))
    else:
        choice=next(r for r in candidates() if recipe_key(r)==best)
    table=pd.DataFrame(rows)
    table['selected']=table.key.eq(recipe_key(choice))
    return dict(choice),table.sort_values('log loss').reset_index(drop=True)


def outer_predictions(oof):
    pieces=[]; choices=[]
    for year in sorted(oof.season.unique()):
        if year<OUTER_FIRST_SEASON: continue
        tr=oof[oof.season<year]
        if tr.season.nunique()<2: continue
        recipe,table=select_recipe(tr)
        te=oof[oof.season==year].copy()
        te['model_wp']=te['p__'+recipe_key(recipe)]
        te['recipe']=recipe_key(recipe)
        te['selection_train_through_season']=int(tr.season.max())
        pieces.append(te)
        choices.append({'test_season':int(year),'recipe':recipe,'selection_train_through_season':int(tr.season.max()),
                        'selection_scores':table.to_dict('records')})
        print(f'  Outer {year}: recipe selected through {int(tr.season.max())}: {recipe_key(recipe)}')
    if not pieces: raise ValueError('No outer evaluation seasons; extend historical window')
    return pd.concat(pieces,ignore_index=True),choices


def bootstrap_ci(delta):
    d=np.asarray(delta,float); n=len(d)
    if n<2: return np.nan,np.nan
    rng=np.random.default_rng(SEED)
    means=[]
    for _ in range(BOOTSTRAP_REPS): means.append(d[rng.integers(0,n,n)].mean())
    return tuple(map(float,np.quantile(means,[.025,.975])))


def scorecard(df):
    cols={'box-score composite':'model_wp','raw spread-derived market':'market_wp','home-field baseline':'homefield_wp'}
    s=df[df['home won'].isin([0.,1.])].replace([np.inf,-np.inf],np.nan).dropna(subset=list(cols.values()))
    rows=[]
    for label,col in cols.items():
        if s.empty: break
        y,p=s['home won'].to_numpy(float),s[col].to_numpy(float)
        picks=~np.isclose(p,.5,rtol=0,atol=1e-12)
        gain=ll(y,s.market_wp)-ll(y,p); lo,hi=bootstrap_ci(gain)
        rows.append({'source':label,'log loss':float(ll(y,p).mean()),'brier':float(np.mean((p-y)**2)),
            'LL gain vs market':float(gain.mean()),'game-bootstrap CI low':lo,'game-bootstrap CI high':hi,
            'games scored':len(s),'games with pick':int(picks.sum()),
            'correct picks':int(((p[picks]>.5)==y[picks]).sum()),
            'picked winner %':float(100*np.mean((p[picks]>.5)==y[picks])) if picks.any() else np.nan})
    return pd.DataFrame(rows)


def _logit(p):
    p=np.clip(np.asarray(p,float),1e-4,1-1e-4); return np.log(p/(1-p))


def fit_blend(market,composite,y,ridge=1e-4):
    """Logistic fit: y ~ a + b_market*logit(market) + b_composite*logit(composite).
    Starts from the pure market (0, 1, 0); a tiny ridge keeps collinear fits stable."""
    X=np.column_stack([np.ones(len(y)),_logit(market),_logit(composite)]); y=np.asarray(y,float)
    def obj(b):
        eta=X@b
        loss=np.mean(np.logaddexp(0,eta)-y*eta)+.5*ridge*np.sum(b[1:]**2)
        grad=X.T@(expit(eta)-y)/len(y)+ridge*np.r_[0.,b[1:]]
        return loss,grad
    return minimize(obj,np.array([0.,1.,0.]),jac=True,method='L-BFGS-B',options={'maxiter':500}).x


def apply_blend(b,market,composite):
    return expit(b[0]+b[1]*_logit(market)+b[2]*_logit(composite))


def market_blend_diagnostic(oof,outer,folds):
    """Reporting only: does the composite carry information the spread lacks?"""
    o=outer[outer['home won'].isin([0.,1.])&outer.market_wp.notna()&outer.model_wp.notna()]
    y=o['home won'].to_numpy(float); m=o.market_wp.to_numpy(float); c=o.model_wp.to_numpy(float)
    rows=[]
    if len(o)<20: return pd.DataFrame(rows)
    b=fit_blend(m,c,y)
    rng=np.random.default_rng(SEED); n=len(y)
    boots=np.array([fit_blend(m[i],c[i],y[i]) for i in (rng.integers(0,n,n) for _ in range(BOOTSTRAP_REPS))])
    lo,hi=np.quantile(boots,[.025,.975],axis=0)
    for j,term in enumerate(('intercept','market log-odds weight','composite log-odds weight')):
        rows.append({'test':'pooled held-out fit (in-sample weights)','season':'pooled','term':term,
                     'estimate':float(b[j]),'CI low':float(lo[j]),'CI high':float(hi[j]),'games':n})
    gains=[]
    for f in folds:
        year=int(f['test_season']); col='p__'+recipe_key(f['recipe'])
        tr=oof[(oof.season<year)&oof['home won'].isin([0.,1.])&oof.market_wp.notna()]
        te=o[o.season==year]
        if len(tr)<MIN_TRAIN_GAMES or te.empty: continue
        bb=fit_blend(tr.market_wp.to_numpy(float),tr[col].to_numpy(float),tr['home won'].to_numpy(float))
        p=apply_blend(bb,te.market_wp.to_numpy(float),te.model_wp.to_numpy(float))
        yy=te['home won'].to_numpy(float)
        gain=ll(yy,te.market_wp.to_numpy(float))-ll(yy,p); gains.append(gain)
        glo,ghi=bootstrap_ci(gain)
        rows.append({'test':'chronological blend (weights from earlier seasons)','season':str(year),'term':'LL gain vs market',
                     'estimate':float(gain.mean()),'CI low':glo,'CI high':ghi,'games':len(te),
                     'blend market weight':float(bb[1]),'blend composite weight':float(bb[2])})
    if gains:
        g=np.concatenate(gains); glo,ghi=bootstrap_ci(g)
        rows.append({'test':'chronological blend (weights from earlier seasons)','season':'pooled','term':'LL gain vs market',
                     'estimate':float(g.mean()),'CI low':glo,'CI high':ghi,'games':len(g)})
    return pd.DataFrame(rows)


def wilson(k,n,z=1.96):
    if n<=0: return np.nan,np.nan
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(c-h),float(c+h)


def calibration_bands(df):
    """Bin games by each source's home-win forecast on the same band edges and
    compare mean forecast with the realized home-win rate (Wilson 95% interval)."""
    s=df[df['home won'].isin([0.,1.])&df.model_wp.notna()&df.market_wp.notna()]
    y=s['home won'].to_numpy(float)
    edges=np.asarray(CAL_BAND_EDGES,float)
    labels=[f'{100*a:.0f}-{100*b:.0f}%' for a,b in zip(edges[:-1],edges[1:])]
    rows=[]
    for source,col in (('model','model_wp'),('market','market_wp')):
        p=s[col].to_numpy(float)
        idx=np.clip(np.digitize(p,edges[1:-1]),0,len(labels)-1)
        for i,lab in enumerate(labels):
            m=idx==i; n=int(m.sum()); k=float(y[m].sum()); lo,hi=wilson(k,n)
            fc=100*float(p[m].mean()) if n else np.nan; re=100*k/n if n else np.nan
            rows.append({'band':lab,'band_index':i,'source':source,'games':n,'mean forecast %':fc,
                         'realized home win %':re,'realized CI low %':100*lo if n else np.nan,
                         'realized CI high %':100*hi if n else np.nan,'realized minus forecast':re-fc if n else np.nan,
                         'forecast outside CI':bool(n and not (100*lo<=fc<=100*hi))})
    return pd.DataFrame(rows)


def calibration_wide(cal):
    w=cal.pivot(index=['band_index','band'],columns='source',values=['games','mean forecast %','realized home win %'])
    w.columns=[f'{src} {metric}' for metric,src in w.columns]
    order=[f'{src} {m}' for src in ('model','market') for m in ('games','mean forecast %','realized home win %')]
    return w[order].reset_index().drop(columns='band_index')


def calibration_summary(cal):
    out=[]
    for src in ('model','market'):
        c=cal[(cal.source==src)&(cal.games>0)]
        err=float((c.games*c['realized minus forecast'].abs()).sum()/c.games.sum()) if len(c) else np.nan
        out.append(f'{src} {err:.1f} pts ({int(c["forecast outside CI"].sum())} of {len(c)} bands outside the 95% interval)')
    return 'Games-weighted mean |realized - forecast|: '+'; '.join(out)+'.'


def calibration_html(cal,esc):
    if cal is None or cal.empty: return ''
    W,pad=320,38; inner=W-2*pad
    X=lambda v:pad+inner*v/100.; Y=lambda v:pad+inner*(1-v/100.)
    svg=[f'<svg viewBox="0 0 {W} {W}" style="width:100%;max-width:340px;flex:0 0 auto" role="img" '
         'aria-label="Reliability chart: mean forecast against realized home-win rate for model and market">']
    for v in (0,25,50,75,100):
        svg.append(f'<line x1="{X(v):.1f}" y1="{Y(0):.1f}" x2="{X(v):.1f}" y2="{Y(100):.1f}" stroke="var(--track)"/>'
                   f'<line x1="{X(0):.1f}" y1="{Y(v):.1f}" x2="{X(100):.1f}" y2="{Y(v):.1f}" stroke="var(--track)"/>'
                   f'<text x="{X(v):.1f}" y="{Y(0)+14:.1f}" font-size="10" text-anchor="middle" fill="var(--mut)">{v}</text>'
                   f'<text x="{X(0)-6:.1f}" y="{Y(v)+3:.1f}" font-size="10" text-anchor="end" fill="var(--mut)">{v}</text>')
    svg.append(f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(100):.1f}" y2="{Y(100):.1f}" stroke="var(--mut)" stroke-dasharray="4 4"/>'
               f'<text x="{W/2:.0f}" y="{W-6}" font-size="11" text-anchor="middle" fill="var(--mut)">mean forecast, home win %</text>'
               f'<text x="12" y="{W/2:.0f}" font-size="11" text-anchor="middle" fill="var(--mut)" transform="rotate(-90 12 {W/2:.0f})">realized home win %</text>')
    for src,color,fill,dx in (('market','var(--away)','var(--card)',2.),('model','var(--home)','var(--home)',-2.)):
        c=cal[(cal.source==src)&(cal.games>0)].sort_values('band_index')
        pts=' '.join(f'{X(r["mean forecast %"])+dx:.1f},{Y(r["realized home win %"]):.1f}' for _,r in c.iterrows())
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" opacity=".7"/>')
        for _,r in c.iterrows():
            x=X(r['mean forecast %'])+dx; rad=min(2.5+np.sqrt(r.games)/3.,8.)
            svg.append(f'<line x1="{x:.1f}" y1="{Y(r["realized CI low %"]):.1f}" x2="{x:.1f}" y2="{Y(r["realized CI high %"]):.1f}" stroke="{color}" opacity=".6"/>'
                       f'<circle cx="{x:.1f}" cy="{Y(r["realized home win %"]):.1f}" r="{rad:.1f}" fill="{fill}" stroke="{color}" stroke-width="1.5">'
                       f'<title>{esc(src)} {esc(r.band)}: {int(r.games)} games, forecast {r["mean forecast %"]:.1f}%, realized {r["realized home win %"]:.1f}%</title></circle>')
    svg.append('</svg>')
    def cell(r):
        if not r.games: return '<td class="num">0</td><td class="num">—</td><td class="num">—</td>'
        cls=' bad' if r['forecast outside CI'] else ''
        return (f'<td class="num">{int(r.games)}</td><td class="num">{r["mean forecast %"]:.1f}%</td>'
                f'<td class="num{cls}">{r["realized home win %"]:.1f}% <span class="mut">({r["realized CI low %"]:.0f}-{r["realized CI high %"]:.0f})</span></td>')
    trs=[]
    for _,g in cal.groupby('band_index'):
        m=g[g.source=='model'].iloc[0]; k=g[g.source=='market'].iloc[0]
        trs.append(f'<tr><td>{esc(m.band)}</td>{cell(m)}{cell(k)}</tr>')
    table=('<div class="slate" style="flex:1 1 360px;margin:0"><table><thead>'
           '<tr><th></th><th colspan="3">Model</th><th colspan="3">Market</th></tr>'
           '<tr><th>Home-win band</th><th class="num">Games</th><th class="num">Forecast</th><th class="num">Realized (95%)</th>'
           '<th class="num">Games</th><th class="num">Forecast</th><th class="num">Realized (95%)</th></tr></thead>'
           f'<tbody>{"".join(trs)}</tbody></table></div>')
    games=int(cal[cal.source=="model"].games.sum())
    return ('<details open><summary>Calibration in matched bands (held-out seasons)</summary>'
            f'<p class="mut">All {games} held-out games are placed in the same home-win bands twice: once by the model\u2019s forecast, once by the market\u2019s. '
            'Within each band, compare the average forecast with how often the home team actually won; whiskers and parentheses are 95% intervals, '
            'and red marks a forecast outside that interval. Points on the dashed diagonal are well calibrated.</p>'
            '<div class="key"><span><i class="sw" style="background:var(--home)"></i>Model</span>'
            '<span><i class="sw" style="background:var(--card);border:1.5px solid var(--away)"></i>Market</span></div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;margin-top:10px">{"".join(svg)}{table}</div>'
            f'<p class="mut">{esc(calibration_summary(cal))}</p></details>')


def record_text(w,l):
    n=w+l
    return f'{w}-{l} ({w/n:.3f})'.replace('(0.','(.') if n else '0-0'


def pick_view(df,col):
    """Pick = the side the source makes the favorite; 50% exactly is no pick."""
    p=df[col].to_numpy(float); y=df['home won'].to_numpy(float)
    conf=np.maximum(p,1-p); pick=np.isfinite(p)&~np.isclose(p,.5)
    won=(p>.5)==(y==1.)
    idx=np.clip(np.digitize(conf,PICK_BAND_EDGES[1:-1]),0,len(PICK_BAND_LABELS)-1)
    return conf,pick,won,idx


def pick_band_index(p):
    conf=max(p,1-p)
    return int(np.clip(np.digitize([conf],PICK_BAND_EDGES[1:-1])[0],0,len(PICK_BAND_LABELS)-1))


def band_records(hist):
    """W-L of each source's picks by pick-confidence band (same edges for both)."""
    h=hist[hist['home won'].isin([0.,1.])&hist.model_wp.notna()&hist.market_wp.notna()]
    rows=[]
    for src,col in (('model','model_wp'),('market','market_wp')):
        conf,pick,won,idx=pick_view(h,col)
        for i,lab in list(enumerate(PICK_BAND_LABELS))+[(len(PICK_BAND_LABELS),'All picks')]:
            m=pick&((idx==i) if i<len(PICK_BAND_LABELS) else True)
            n=int(m.sum()); w=int(won[m].sum()); lo,hi=wilson(w,n)
            rows.append({'source':src,'band_index':i,'band':lab,'wins':w,'losses':n-w,'games':n,
                         'record':record_text(w,n-w),
                         'avg pick prob %':100*float(conf[m].mean()) if n else np.nan,
                         'realized pick win %':100*w/n if n else np.nan,
                         'CI low %':100*lo if n else np.nan,'CI high %':100*hi if n else np.nan})
    return pd.DataFrame(rows)


def band_records_wide(recs):
    w=recs.pivot(index=['band_index','band'],columns='source',values=['record','avg pick prob %','realized pick win %'])
    w.columns=[f'{src} {m}' for m,src in w.columns]
    order=[f'{src} {m}' for src in ('model','market') for m in ('record','avg pick prob %','realized pick win %')]
    out=w[order].reset_index().drop(columns='band_index')
    for c in out.columns:
        if c.endswith('%'): out[c]=pd.to_numeric(out[c]).round(1)
    return out


def records_source_label(hist,season,cur):
    held=hist[hist.season<season]; now=hist[hist.season==season]
    txt=f'{len(hist)} games: held-out {int(held.season.min())}-{int(held.season.max())}' if len(held) else f'{len(hist)} games'
    if len(now): txt+=f' plus {season} weeks {int(now.week.min())}-{int(now.week.max())}'
    return txt+', all walk-forward reconstructions'


def _rec_cell(r,cls=''):
    if not r.games: return '<td class="num">0-0</td><td class="num">—</td><td class="num">—</td>'
    bad=' bad' if not (r['CI low %']<=r['avg pick prob %']<=r['CI high %']) else ''
    return (f'<td class="num{cls}"><b>{esc_(r.record)}</b></td><td class="num">{r["avg pick prob %"]:.1f}%</td>'
            f'<td class="num{bad}">{r["realized pick win %"]:.1f}% <span class="mut">({r["CI low %"]:.0f}-{r["CI high %"]:.0f})</span></td>')


def esc_(x): return escape(str(x),quote=True)


def records_section_html(recs,label):
    if recs is None or recs.empty: return ''
    trs=[]
    for i,g in recs.groupby('band_index'):
        m=g[g.source=='model'].iloc[0]; k=g[g.source=='market'].iloc[0]
        style=' style="border-top:2px solid var(--rule);font-weight:600"' if m.band=='All picks' else ''
        trs.append(f'<tr{style}><td>{esc_(m.band)}</td>{_rec_cell(m)}{_rec_cell(k)}</tr>')
    return ('<details open><summary>Realized records in matched bands</summary>'
            f'<p class="mut">Each source picks the side it makes the favorite. Picks are grouped by how confident they were, using the same bands for model and market. '
            f'Avg is the average pick probability in the band; realized is how often those picks won, with a 95% interval (red when the interval misses the average). '
            f'Records: {esc_(label)}. The current week never counts toward its own records. No wager recommendations.</p>'
            '<div class="slate"><table><thead><tr><th></th><th colspan="3">Model picks</th><th colspan="3">Market picks</th></tr>'
            '<tr><th>Pick band</th><th class="num">Record</th><th class="num">Avg</th><th class="num">Realized (95%)</th>'
            '<th class="num">Record</th><th class="num">Avg</th><th class="num">Realized (95%)</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div></details>')


def card_records_html(r,recs):
    """One game card's matched-band records: the model's record in the model's band
    for this game and the market's record in the market's band."""
    if recs is None or recs.empty: return ''
    trs=[]
    for src,col,name in (('model','model_wp','Model'),('market','market_wp','Market')):
        p=_num(r.get(col))
        if not np.isfinite(p): trs.append(f'<tr><td>{name}</td><td colspan="4" class="mut">no price</td></tr>'); continue
        if np.isclose(p,.5): trs.append(f'<tr><td>{name}</td><td colspan="4" class="mut">no pick (50%)</td></tr>'); continue
        team=r['home'] if p>.5 else r['away']; i=pick_band_index(p)
        b=recs[(recs.source==src)&(recs.band_index==i)].iloc[0]
        real=(f'{b["realized pick win %"]:.1f}% <span class="mut">({b["CI low %"]:.0f}-{b["CI high %"]:.0f})</span>' if b.games else '—')
        trs.append(f'<tr><td>{name}</td><td><b>{esc_(team)}</b> {100*max(p,1-p):.0f}%</td><td class="num">{esc_(b.band)}</td>'
                   f'<td class="num"><b>{esc_(b.record)}</b></td><td class="num">{real}</td></tr>')
    return ('<h3>Realized record in this game\u2019s band</h3>'
            '<div class="slate recs"><table><thead><tr><th></th><th>Pick</th><th class="num">Band</th>'
            '<th class="num">Band record</th><th class="num">Realized (95%)</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def blend_verdict(blend):
    if blend.empty: return 'Market-blend check: not enough held-out games.'
    w=blend[blend.term=='composite log-odds weight'].iloc[0]
    if w['CI low']>0: v='the composite adds information the spread lacks'
    elif w['CI high']<0: v='the composite enters with a NEGATIVE weight (it mostly echoes the spread with noise)'
    else: v='no reliable evidence the composite adds anything beyond the spread'
    out=f"Composite weight {w.estimate:+.3f} (95% {w['CI low']:+.3f} to {w['CI high']:+.3f}): {v}."
    g=blend[(blend.term=='LL gain vs market')&(blend.season=='pooled')]
    if len(g):
        g=g.iloc[0]
        out+=(f" Chronological blend vs market: {g.estimate:+.4f} log loss per game "
              f"(95% {g['CI low']:+.4f} to {g['CI high']:+.4f}; positive means the blend beat the market).")
    return out


def current_predictions(features,recipe,year,through_week):
    f=features[recipe['half_life']]; rows=[]; current_fit=None
    for week in range(1,through_week+1):
        te=f[(f.season==year)&(f.week==week)].copy()
        if te.empty: continue
        tr=f[before(f,year,week)]
        fit=fit_composite(tr,recipe,year,week)
        baseline=fit_composite(tr,recipe,year,week,baseline=True)
        p,z,_=apply_fit(te,fit,True)
        te['model_wp'],te['composite_log_odds']=p,z
        te['homefield_wp']=apply_fit(te,baseline)
        te['recipe']=recipe_key(recipe)
        rows.append(te)
        if week==through_week: current_fit=fit
    return pd.concat(rows,ignore_index=True),current_fit


def coefficient_table(fit):
    return pd.DataFrame({'feature':fit['names'],'label':[feature_label(n) for n in fit['names']],
                         'coefficient per scaled unit':fit['beta'],
                         'training scale':fit['scale'],
                         'training missing fraction':[fit['training_missing_fractions'][n] for n in fit['names']]})


def contribution_table(board,fit):
    _,_,parts=apply_fit(board,fit,True)
    rows=[]
    for i,r in enumerate(board.to_dict('records')):
        for j,name in enumerate(fit['names']):
            rows.append({'game_id':r['game_id'],'game':r['away']+'@'+r['home'],'feature':name,
                         'home-minus-away input':r[name] if name!='site' else r['site'],
                         'log-odds contribution':float(parts[i,j]),
                         'direction':'home' if parts[i,j]>0 else 'away' if parts[i,j]<0 else 'neutral'})
    return pd.DataFrame(rows)

# Frozen recipe and genuine pre-kickoff snapshots.
def config_signature():
    cfg={'revision':REVISION,'candidates':candidates(),'fit_half_life_seasons':FIT_HALF_LIFE_SEASONS,
         'offseason_retention':OFFSEASON_RETENTION,'prior_games':PRIOR_EQUIVALENT_GAMES,
         'max_history_seasons':MAX_HISTORY_SEASONS,'min_history':MIN_TEAM_HISTORY,
         'min_train':MIN_TRAIN_GAMES,'one_se':SELECTION_ONE_SE,'selection_rule':SELECTION_RULE,
         'families':list(FEATURE_FAMILIES),'rates_core':RATES_CORE,'backtest_first':BACKTEST_FIRST_SEASON,'warmup_seasons':WARMUP_SEASONS,
         'redundant_allowed':sorted(REDUNDANT_ALLOWED),'adjusted_families':ADJUSTED_FAMILIES,
         'avail':{'families':AVAIL_FAMILIES,'status_weight':STATUS_WEIGHT,'roster_out':ROSTER_OUT,'window':AVAIL_WINDOW,
                  'qb_half_life':QB_HALF_LIFE,'qb_prior':QB_PRIOR_DROPBACKS,'pos_group':POS_GROUP,
                  'roster_membership':ROSTER_MEMBERSHIP,'membership_max_share':MEMBERSHIP_MAX_SHARE,
                  'family_cols':AVAIL_FAMILY_COLS,'cs_min_games':CS_MIN_GAMES,'qb_rule':QB_RULE},
         'adjust_age':'league week-slots','rates':RATES,'totals':TOTALS}
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),json.loads(json.dumps(cfg))


def frozen_recipe(oof,season):
    sig,cfg=config_signature()
    path=OUTPUT_ROOT/f'frozen_recipe_{season}.json'
    if path.exists():
        rec=json.loads(path.read_text())
        if rec['config_signature']!=sig:
            raise ValueError(f'{path} belongs to a different recipe configuration. Use a NEW OUTPUT_ROOT to start a separate experiment; the existing record is preserved.')
        if rec['selection_max_season']>=season: raise ValueError('Frozen recipe has future training data')
        print(f'  Using frozen {season} recipe: {recipe_key(rec["recipe"])}')
        return rec
    tr=oof[oof.season<season]
    recipe,table=select_recipe(tr)
    rec={'revision':REVISION,'created_utc':now_utc().isoformat(),'season':int(season),'recipe':recipe,
         'config_signature':sig,'config':cfg,'selection_max_season':int(tr.season.max()),
         'selection_data_sha256':data_hash(tr),'selection_scores':table.to_dict('records'),
         'freeze_scope':'Hyperparameters and feature recipe fixed; regression coefficients refit each week on earlier games.'}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(rec,f,indent=2,allow_nan=False)
    return rec


def kickoff_utc(row):
    try:
        naive=dt.datetime.fromisoformat(str(row['gameday'])+'T'+str(row['gametime']))
        return naive.replace(tzinfo=ZoneInfo('America/New_York')).astimezone(dt.timezone.utc)
    except (ValueError,TypeError): return None


def record_forward(board,fit,recipe_record,asof=None):
    """First pre-kickoff snapshot per game/recipe only. Final/in-progress rows
    cannot be backfilled as forecasts. Unknown kickoff times are skipped.
    The ledger is local to OUTPUT_ROOT; it is not a trusted external timestamp.
    """
    asof=now_utc() if asof is None else asof
    path=OUTPUT_ROOT/'forward_predictions.jsonl'
    old=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    sig=recipe_record['config_signature']
    identity=f'{sig}:{recipe_key(recipe_record["recipe"])}:{recipe_record["season"]}'
    seen={(r['experiment'],r['game_id']) for r in old}
    new=[]
    for r in board.to_dict('records'):
        ko=kickoff_utc(r)
        if pd.notna(r['result']) or ko is None or ko<=asof or not r['ready']: continue
        if (identity,r['game_id']) in seen: continue
        record={'experiment':identity,'revision':REVISION,'game_id':r['game_id'],'season':int(r['season']),
            'week':int(r['week']),'home':r['home'],'away':r['away'],'generated_utc':asof.isoformat(),
            'kickoff_utc':ko.isoformat(),'model_wp':float(r['model_wp']),
            'market_wp':float(r['market_wp']) if pd.notna(r['market_wp']) else None,
            'homefield_wp':float(r['homefield_wp']),
            'spread_line':float(r['spread_line']) if pd.notna(r['spread_line']) else None,
            'recipe':recipe_record['recipe'],'coefficient_fit':fit,
            'source_note':'First locally recorded pre-kickoff forecast; source market quote has no independent quote timestamp.'}
        new.append(record); seen.add((identity,r['game_id']))
    if new:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as f:
            for r in new: f.write(json.dumps(r,allow_nan=False)+'\n')
    return old+new,len(new)


def grade_forward(records,schedules):
    if not records: return pd.DataFrame(),pd.DataFrame()
    r=pd.DataFrame([{k:v for k,v in x.items() if k!='coefficient_fit'} for x in records])
    for x in records:
        if dt.datetime.fromisoformat(x['generated_utc'])>=dt.datetime.fromisoformat(x['kickoff_utc']):
            raise ValueError('Forward ledger contains a non-pregame timestamp')
    s=r.merge(schedules[['game_id','result','home won']],on='game_id',how='left',validate='many_to_one')
    tables=[]
    for name,t in s.groupby('experiment'):
        card=scorecard(t)
        if len(card): tables.append(card.assign(experiment=name))
    return s,pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()

# Board rendering.
def feature_label(name):
    if name=='site': return 'Home field'
    if name.startswith('d__avail__'): return 'Availability: '+AVAIL_LABEL.get(name[10:],name[10:])
    _,role,_,metric=name.split('__',3)
    m=METRIC_LABEL.get(metric,metric)
    return f'Offense: {m}' if role=='for' else f'Defense: opponents\u2019 {m}'


def profile_label(role,metric):
    m=METRIC_LABEL.get(metric,metric)
    return m if role=='for' else f'opponents\u2019 {m}'


def _num(x):
    try: x=float(x)
    except (TypeError,ValueError): return np.nan
    return x


def better_side(role,metric,away,home):
    hb=OFFENSE_HIGHER_BETTER.get(metric)
    if hb is None or not (np.isfinite(away) and np.isfinite(home)) or np.isclose(away,home): return None
    if role=='allowed': hb=not hb
    return 'home' if (home>away)==hb else 'away'


BOARD_CSS="""
.nflb{--paper:#f6f7f4;--card:#fff;--ink:#16241d;--mut:#5d6d65;--rule:#d8ddd6;--track:#e7ebe5;
 --home:#2457a6;--away:#c0601e;--good:#1d7a3c;--bad:#b4442a;--win:#e9f1e6;
 font:15px/1.5 "Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
 background:var(--paper);padding:20px clamp(12px,3vw,28px);max-width:980px;margin:0 auto;border-radius:14px}
@media (prefers-color-scheme:dark){.nflb{--paper:#121a16;--card:#1a241f;--ink:#e5ece7;--mut:#9aa9a1;
 --rule:#2c3832;--track:#26322c;--home:#6fa2f0;--away:#eb9a5c;--good:#58c27d;--bad:#ef7f62;--win:#20352a}}
.nflb *{box-sizing:border-box}
.nflb h1,.nflb h2,.nflb .team{font-family:"Barlow Condensed","Arial Narrow",system-ui,sans-serif;font-weight:700;letter-spacing:.01em}
.nflb h1{font-size:34px;line-height:1.05;margin:0 0 4px}
.nflb h2{font-size:28px;line-height:1;margin:0}
.nflb h3{font-size:15px;margin:18px 0 6px}
.nflb p{margin:6px 0;max-width:72ch}
.nflb .mut{color:var(--mut);font-size:13.5px}
.nflb .num{text-align:right;font-variant-numeric:tabular-nums}
.nflb .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:16px 0}
.nflb .tile{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px 14px}
.nflb .tile .k{font-weight:600}
.nflb .tile .v{font:700 30px/1.1 "Barlow Condensed","Arial Narrow",sans-serif;margin:4px 0}
.nflb .good{color:var(--good)}.nflb .bad{color:var(--bad)}
.nflb .slate{background:var(--card);border:1px solid var(--rule);border-radius:10px;overflow-x:auto;margin:16px 0}
.nflb table{width:100%;border-collapse:collapse;font-size:14px}
.nflb th{font-weight:600;text-align:left;color:var(--mut);padding:8px 10px;border-bottom:1px solid var(--rule)}
.nflb td{padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:middle}
.nflb tr:last-child td{border-bottom:0}
.nflb a{color:inherit}
.nflb a:focus-visible,.nflb summary:focus-visible{outline:2px solid var(--home);outline-offset:2px}
.nflb .wp{display:flex;align-items:center;gap:8px;min-width:120px}
.nflb .wp .bar{flex:1;height:8px;border-radius:4px;background:var(--away);overflow:hidden}
.nflb .wp .bar i{display:block;height:100%;background:var(--home);margin-left:auto}
.nflb .flag{color:var(--bad);font-weight:700}
.nflb .tag{display:inline-block;font-size:12.5px;padding:1px 8px;border-radius:999px;border:1px solid var(--rule);color:var(--mut)}
.nflb article{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:16px 18px;margin:14px 0}
.nflb article header{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}
.nflb .at{color:var(--mut);font-weight:400}
.nflb .split{display:flex;height:34px;border-radius:6px;overflow:hidden;margin:12px 0 4px}
.nflb .split .a{background:var(--away)}.nflb .split .h{background:var(--home)}
.nflb .split span{display:flex;align-items:center;padding:0 10px;color:#fff;font:700 18px "Barlow Condensed","Arial Narrow",sans-serif;white-space:nowrap;min-width:0;overflow:hidden}
.nflb .split .h{justify-content:flex-end}
.nflb .final{font-weight:600}
.nflb .ok{color:var(--good)}.nflb .miss{color:var(--bad)}
.nflb .track{position:relative;height:10px;background:var(--track);border-radius:5px;min-width:110px}
.nflb .track::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--mut)}
.nflb .track i{position:absolute;top:0;bottom:0}
.nflb .track .pos{background:var(--home);border-radius:0 5px 5px 0}
.nflb .track .neg{background:var(--away);border-radius:5px 0 0 5px}
.nflb td.win{background:var(--win);font-weight:600}
.nflb details{margin-top:12px}
.nflb summary{cursor:pointer;font-weight:600}
.nflb .key{display:flex;gap:16px;flex-wrap:wrap;font-size:13.5px;color:var(--mut)}
.nflb .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.nflb .games td:first-child{white-space:nowrap}.nflb .games td:last-child{min-width:150px}
@media (max-width:600px){.nflb .wp .bar{display:none}.nflb .wp{min-width:0}.nflb .slate td,.nflb .slate th{padding:7px 6px}.nflb h1{font-size:28px}}
"""


def _fmt_avail(c,v):
    v=_num(v)
    if not np.isfinite(v): return '—'
    return f'{v:+.2f}' if c=='qb_delta' else f'{100*v:.0f}%'


def html_board(board,fit,outer_card,recipe_record,generated,season_card=None,week_card=None,calib=None,records=None):
    esc=lambda x:escape(str(x),quote=True)
    recipe=recipe_record['recipe']; fam=recipe['family']
    pfam=profile_family(fam)  # family whose stat profiles are shown
    has_avail=fam in AVAIL_FAMILIES
    parts=contribution_table(board,fit)
    cmax=max(float(parts['log-odds contribution'].abs().max()),1e-9) if len(parts) else 1.
    tz=ZoneInfo(TIMEZONE)
    far=dt.datetime.max.replace(tzinfo=dt.timezone.utc)
    rows=board.to_dict('records')
    rows.sort(key=lambda r:(pd.notna(r['result']),kickoff_utc(r) or far,str(r['game_id'])))

    def when(r):
        ko=kickoff_utc(r)
        if ko is None: return f'{r["gameday"]} {r["gametime"]} ET'
        k=ko.astimezone(tz)
        return f'{k:%a %b} {k.day}, {k.hour%12 or 12}:{k:%M} {"AM" if k.hour<12 else "PM"} {k:%Z}'

    def mark(q,y):
        if not np.isfinite(q) or y==.5 or np.isclose(q,.5): return '<span class="mut">no pick</span>'
        return '<span class="ok">right</span>' if (q>.5)==(y==1.) else '<span class="miss">wrong</span>'

    def score(r):
        a,h=_num(r.get('away_score')),_num(r.get('home_score'))
        if np.isfinite(a) and np.isfinite(h): return f'{r["away"]} {a:.0f}, {r["home"]} {h:.0f}'
        return f'Home margin {_num(r["result"]):+.0f}'

    def tile(card,label,ci=False):
        try:
            c=card.set_index('source'); m=c.loc['box-score composite']; k=c.loc['raw spread-derived market']
        except (AttributeError,KeyError): return ''
        g=float(m['LL gain vs market'])
        extra=(f'<div class="mut">95% game-bootstrap range {m["game-bootstrap CI low"]:+.3f} to {m["game-bootstrap CI high"]:+.3f}</div>'
               if ci and np.isfinite(m['game-bootstrap CI low']) else '')
        return (f'<div class="tile"><div class="k">{esc(label)}</div>'
                f'<div class="v {"good" if g>0 else "bad"}">{g:+.3f}</div>'
                f'<div class="mut">Log-loss edge vs market over {int(m["games scored"])} games (positive means the model beat it)</div>'
                f'<div class="mut">Winners picked: model {int(m["correct picks"])} of {int(m["games with pick"])}, market {int(k["correct picks"])} of {int(k["games with pick"])}</div>{extra}</div>')

    gap_note=('Large gap from the market. Availability here uses final injury reports and the most recent roster only, '
              'not game-day inactives or late news, so check those first.' if has_avail else
              'Large gap from the market. The model has no injury, quarterback or roster news, so check that first.')

    out=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>NFL box-score composite</title>',
         '<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">',
         f'<style>{BOARD_CSS}</style></head><body><div class="nflb">']
    seasons=sorted(board.season.unique()); weeks=sorted(board.week.unique())
    out+=[f'<h1>Week {", ".join(map(str,weeks))} win probabilities</h1>',
          f'<p class="mut">{esc(seasons[0]) if seasons else ""} season. Generated {esc(generated)}. '
          f'Recipe {esc(recipe_key(recipe))}: {esc(fam)} features, {recipe["half_life"]:g}-game team half-life, ridge {recipe["ridge"]:g}, '
          f'chosen from seasons through {recipe_record["selection_max_season"]}. Coefficients refit each week on earlier games only.</p>',
          '<div class="key"><span><i class="sw" style="background:var(--away)"></i>Away</span>'
          '<span><i class="sw" style="background:var(--home)"></i>Home</span>'
          f'<span><b class="flag">!</b> model and market differ by {FLAG_GAP_PP:g}+ points</span></div>',
          '<div class="tiles">',
          tile(outer_card,'Held-out seasons (reconstructed)',ci=True),
          tile(season_card,'This season so far'),
          tile(week_card,'This week'),'</div>']

    slate=['<div class="slate"><table class="games"><thead><tr><th>Game</th><th>Model home win %</th><th class="num">Market</th>'
           '<th class="num">Gap</th><th>Result</th></tr></thead><tbody>']
    for r in rows:
        p=float(r['model_wp']); m=_num(r['market_wp']); gap=100*(p-m) if np.isfinite(m) else np.nan
        flagged=np.isfinite(gap) and abs(gap)>=FLAG_GAP_PP
        if pd.isna(r['result']): res='<span class="tag">Upcoming</span>'
        else:
            y=won(r['result'])
            res=f'{esc(score(r))}<br><span class="mut">Model </span>{mark(p,y)}<span class="mut">, market </span>{mark(m,y)}'
        slate.append(f'<tr><td><a href="#g-{esc(r["game_id"])}"><b>{esc(r["away"])} at {esc(r["home"])}</b></a>'
                     f'<br><span class="mut">{esc(when(r))}</span></td>'
                     f'<td><div class="wp"><div class="bar"><i style="width:{100*p:.1f}%"></i></div><b>{100*p:.0f}%</b></div></td>'
                     f'<td class="num">{"—" if not np.isfinite(m) else f"{100*m:.0f}%"}</td>'
                     f'<td class="num{" flag" if flagged else ""}">{"—" if not np.isfinite(gap) else f"{gap:+.1f}"}{" !" if flagged else ""}</td>'
                     f'<td>{res}</td></tr>')
    slate.append('</tbody></table></div>')
    out+=slate
    recs,rec_label=records if records is not None else (None,'')
    out.append(records_section_html(recs,rec_label))
    out.append(calibration_html(calib,esc))

    for r in rows:
        p=float(r['model_wp']); m=_num(r['market_wp']); gap=100*(p-m) if np.isfinite(m) else np.nan
        neutral=_num(r['site'])==0
        status='<span class="tag">Upcoming, prior-game inputs</span>' if pd.isna(r['result']) else '<span class="tag">Final, reconstructed backtest</span>'
        out+=[f'<article id="g-{esc(r["game_id"])}"><header><h2>{esc(r["away"])} <span class="at">at</span> {esc(r["home"])}</h2>{status}</header>',
              f'<p class="mut">{esc(when(r))}{", neutral site" if neutral else ""}</p>',
              f'<div class="split" role="img" aria-label="Model: {esc(r["away"])} {100*(1-p):.0f} percent, {esc(r["home"])} {100*p:.0f} percent">'
              f'<span class="a" style="width:{100*(1-p):.1f}%">{esc(r["away"])} {100*(1-p):.0f}%</span>'
              f'<span class="h" style="width:{100*p:.1f}%">{esc(r["home"])} {100*p:.0f}%</span></div>']
        line=f'Composite log-odds edge {r["composite_log_odds"]:+.3f} toward {esc(r["home"] if r["composite_log_odds"]>=0 else r["away"])}.'
        if np.isfinite(m):
            line+=f' Market has {esc(r["home"])} at {100*m:.1f}%, so the model is {abs(gap):.1f} points {"higher" if gap>=0 else "lower"} on the home side.'
        out.append(f'<p>{line}</p>')
        if np.isfinite(gap) and abs(gap)>=FLAG_GAP_PP:
            out.append(f'<p class="flag">{gap_note}</p>')
        if pd.notna(r['result']):
            y=won(r['result'])
            out.append(f'<p class="final">Final: {esc(score(r))}. Model {mark(p,y)}, market {mark(m,y)}.</p>')
        out.append(card_records_html(r,recs))
        if not r['ready']:
            out.append('<p class="mut">Limited team history, so this forecast leans on the league prior and is kept out of the forward ledger.</p>')
        t=parts[parts.game_id==r['game_id']].copy()
        t=t.loc[t['log-odds contribution'].abs().sort_values(ascending=False).index].head(6)
        out+=['<h3>Largest score contributions</h3>',
              f'<p class="mut">Bars to the left push toward {esc(r["away"])}, to the right toward {esc(r["home"])}. Units are log-odds; features overlap, so read these as the model\u2019s bookkeeping, not causes.</p>',
              '<div class="slate"><table><tbody>']
        for c in t.to_dict('records'):
            v=float(c['log-odds contribution']); w=50*abs(v)/cmax
            style=f'left:50%;width:{w:.1f}%' if v>0 else f'right:50%;width:{w:.1f}%'
            team=r['home'] if v>0 else r['away'] if v<0 else 'neither'
            out.append(f'<tr><td>{esc(feature_label(c["feature"]))}</td>'
                       f'<td style="width:38%"><div class="track"><i class="{"pos" if v>0 else "neg"}" style="{style}"></i></div></td>'
                       f'<td class="num">{v:+.3f}</td><td>{esc(team)}</td></tr>')
        out.append('</tbody></table></div>')
        prow=[]
        for role,title in (('for','Offense'),('allowed','Defense')):
            prow.append(f'<tr><th colspan="3">{title}</th></tr>')
            for metric in FAMILIES[pfam]:
                key=f'{role}__{pfam}__{metric}'
                a,h=_num(r.get('away__'+key)),_num(r.get('home__'+key))
                side=better_side(role,metric,a,h)
                fa=f'{a:.2f}' if np.isfinite(a) else '—'; fh=f'{h:.2f}' if np.isfinite(h) else '—'
                prow.append(f'<tr><td>{esc(profile_label(role,metric))}</td>'
                            f'<td class="num{" win" if side=="away" else ""}">{fa}</td>'
                            f'<td class="num{" win" if side=="home" else ""}">{fh}</td></tr>')
        if has_avail:
            arow=[]
            for side in ('away','home'):
                exp,usual=r.get(f'{side}_qb_expected'),r.get(f'{side}_qb_usual')
                if isinstance(exp,str) and isinstance(usual,str) and exp!=usual:
                    arow.append(f'<p class="flag">{esc(r[side])}: {esc(exp)} expected at quarterback instead of {esc(usual)}.</p>')
            trs=''.join(f'<tr><td>{esc(AVAIL_LABEL[c])}</td>'
                        f'<td class="num">{_fmt_avail(c,r.get("away__avail__"+c))}</td><td class="num">{_fmt_avail(c,r.get("home__avail__"+c))}</td></tr>'
                        for c in AVAIL_FAMILY_COLS[fam])
            out+=['<h3>Availability</h3>',*arow,
                  '<p class="mut">The QB row compares the projected starter with the team\u2019s recent dropback mix, so it is nonzero even without a change. '
                  'Unit rows count players listed Out/Doubtful (Questionable as a quarter), moved off the roster, or no longer on it'
                  +(f'; once a team has {CS_MIN_GAMES} games this season, only this season\u2019s games count' if fam.endswith('_cs') else '')+'.</p>',
                  f'<div class="slate"><table><thead><tr><th>Injury report and most recent roster</th><th class="num">{esc(r["away"])}</th>'
                  f'<th class="num">{esc(r["home"])}</th></tr></thead><tbody>{trs}</tbody></table></div>']
        out+=['<details><summary>Team stat profiles</summary>',
              f'<p class="mut">{"Opponent-adjusted prior-game ratings: what each unit would post against a league-average opponent." if pfam in ADJUSTED_FAMILIES else "Decayed prior-game averages, not adjusted for opponents."} The shaded cell is the better side for that stat; plays per game is left neutral.</p>',
              f'<div class="slate"><table><thead><tr><th>Stat</th><th class="num">{esc(r["away"])}</th><th class="num">{esc(r["home"])}</th></tr></thead><tbody>',
              *prow,'</tbody></table></div></details></article>']
    note=('Availability uses final injury-report status and the most recent roster before the game (status and membership; '
          'players no longer on the roster count as out; only rostered quarterbacks are projected); game-day inactives are not used. '
          if has_avail else 'No live injury/starter adjustment. ')
    note+=('Profiles are opponent-adjusted. ' if pfam in ADJUSTED_FAMILIES else 'No explicit opponent-strength adjustment. ')
    out.append(f'<p class="mut">{note}Full-game box-score history includes game-state effects. '
               'Penalties are reconstructed penalized-play proxies; possession time is source drive metadata with coverage checks. Missing stats use a neutral matchup contribution. '
               'Past results are reconstructions, not archived forecasts. Game-bootstrap ranges omit model-selection uncertainty and serial team dependence. No wager recommendations.</p>')
    out.append('</div></body></html>')
    return ''.join(out)


def definitions_table():
    rows=[]
    for family,specs in FAMILIES.items():
        for name,(num,den,mult) in specs.items():
            rows.append({'family':family,'feature':name,'numerator':num,'denominator':den or 'games',
                         'multiplier':mult,'profiles':'own offense and opposing production allowed',
                         'in_regression':'offense only' if name in REDUNDANT_ALLOWED else 'offense and defense',
                         'opponent_adjusted':profile_family(family) in ADJUSTED_FAMILIES,
                         'timing':'strictly prior weeks; decayed counts with league-prior smoothing'})
    return pd.DataFrame(rows)


def cached_grid(features,season):
    sig,_=config_signature()
    h=hashlib.sha256(sig.encode())
    for half in TEAM_HALF_LIVES:
        hist=features[half]; hist=hist[hist.season<season]
        h.update(data_hash(hist).encode())
    path=OUTPUT_ROOT/'cache'/f'walk_forward_{h.hexdigest()[:16]}.csv'
    meta=path.with_suffix('.json')
    if REUSE_CACHE and path.exists() and meta.exists():
        m=json.loads(meta.read_text())
        if m['sha256']==hashlib.sha256(path.read_bytes()).hexdigest():
            print('Reusing matching historical walk-forward predictions.')
            return pd.read_csv(path)
    oof=walk_forward_grid(features,season-1)
    path.parent.mkdir(parents=True,exist_ok=True); oof.to_csv(path,index=False)
    dump(meta,{'revision':REVISION,'generated_utc':now_utc().isoformat(),
               'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_hash':h.hexdigest()})
    return oof


def _ledger_key(line):
    r=json.loads(line); return (r['experiment'],r['game_id'])


def migrate_local_outputs(local,root):
    """Copy an earlier local OUTPUT folder into root without overwriting anything
    already there. Forward-ledger records are merged by (experiment, game_id);
    any other file that exists in both places keeps the root (Drive) version."""
    local,root=Path(local),Path(root)
    report={'copied':0,'kept_existing':0,'ledger_records_added':0}
    if not local.is_dir() or local.resolve()==root.resolve(): return report
    for src in sorted(local.rglob('*')):
        if not src.is_file() or src.name=='.migrated_to_drive': continue
        rel=src.relative_to(local); dst=root/rel
        if rel.as_posix()=='forward_predictions.jsonl' and dst.exists():
            have={_ledger_key(l) for l in dst.read_text().splitlines() if l.strip()}
            add=[l for l in src.read_text().splitlines() if l.strip() and _ledger_key(l) not in have]
            if add:
                with dst.open('a',encoding='utf-8') as f:
                    for l in add: f.write(l+'\n')
            report['ledger_records_added']+=len(add)
        elif dst.exists():
            if dst.read_bytes()!=src.read_bytes(): report['kept_existing']+=1
        else:
            dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst); report['copied']+=1
    (local/'.migrated_to_drive').write_text(now_utc().isoformat(),encoding='utf-8')
    return report


def resolve_output_root():
    """Google Drive in Colab (mounting it if needed); the local folder otherwise."""
    local=Path(OUTPUT_NAME)
    if not USE_GOOGLE_DRIVE: return local
    try:
        from google.colab import drive
    except ImportError:
        return local
    if not (Path(DRIVE_MOUNT)/'MyDrive').exists(): drive.mount(DRIVE_MOUNT)
    root=Path(DRIVE_MOUNT)/'MyDrive'/DRIVE_FOLDER/OUTPUT_NAME
    root.mkdir(parents=True,exist_ok=True)
    for old in dict.fromkeys([local.resolve(),Path('/content')/OUTPUT_NAME]):
        if old.is_dir() and not (old/'.migrated_to_drive').exists():
            r=migrate_local_outputs(old,root)
            print(f'  Copied earlier local outputs from {old} to Drive: {r["copied"]} files, '
                  f'{r["ledger_records_added"]} forward-ledger records merged, {r["kept_existing"]} Drive versions kept.')
    return root


def main():
    global OUTPUT_ROOT
    for pkg in ('nflreadpy','pyarrow'):
        if importlib.util.find_spec(pkg) is None:
            subprocess.run([sys.executable,'-m','pip','install','-q',pkg],check=True)
    AUDIT.clear()
    OUTPUT_ROOT=resolve_output_root()
    on_drive=str(OUTPUT_ROOT).startswith(DRIVE_MOUNT)
    print(f'Output folder: {OUTPUT_ROOT.resolve()}{" (Google Drive)" if on_drive else ""}')
    now=now_utc(); today=now.astimezone(ZoneInfo(TIMEZONE)).date()
    season=int(SEASON if SEASON is not None else today.year if today.month>=8 else today.year-1)
    years=list(range(BACKTEST_FIRST_SEASON-WARMUP_SEASONS,season+1))
    if season<=OUTER_FIRST_SEASON: raise ValueError('Need at least one completed outer-test season')
    print(f'NFL box-score composite {REVISION} | season {season}')
    print('Predictors: prior-game offense and defense only. No market features or same-game final stats.')
    schedules=pd.concat([load_schedule(y) for y in years],ignore_index=True)
    current=schedules[schedules.season==season]
    pending=current.loc[current.result.isna(),'week']
    cur=int(CURRENT_WEEK if CURRENT_WEEK is not None else pending.min() if len(pending) else current.week.max())
    if cur not in set(current.week): raise ValueError('Requested week is not scheduled')
    schedules=schedules[(schedules.season<season)|(schedules.week<=cur)].copy()
    outdir=OUTPUT_ROOT/f'{season}_week{cur}_{now.strftime("%Y%m%dT%H%M%S%fZ")}'
    outdir.mkdir(parents=True,exist_ok=False)
    saved=[]
    def save(df,name): df.to_csv(outdir/name,index=False); saved.append(name)
    print('[1/4] Loading full-game box-score stat types')
    if TEAM_GAME_CSV:
        box=validate_boxes(normalize_teams(pd.read_csv(TEAM_GAME_CSV)))
        box=box[box.game_id.isin(schedules.loc[schedules.result.notna(),'game_id'])]
        AUDIT.append({'source':'team_game_csv','path':str(TEAM_GAME_CSV),'sha256':data_hash(box)})
        qb=None  # QB layer needs play-by-play passers
    else:
        boxes=[]; qbs=[]
        for y in years:
            print(f'  Box-score reconstruction {y}')
            b,q=load_boxes(y,schedules[schedules.season==y])
            if len(b): boxes.append(b)
            if q is not None and len(q): qbs.append(q)
        if not boxes: raise ValueError('No box-score history available')
        box=validate_boxes(pd.concat(boxes,ignore_index=True))
        qb=pd.concat(qbs,ignore_index=True) if qbs else None
    truth=schedules[['game_id','season','week','home_team','away_team']]
    check=box.merge(truth,on='game_id',how='left',suffixes=('','_schedule'),validate='many_to_one')
    if not ((check.season==check.season_schedule)&(check.week==check.week_schedule)&
            ((check.team==check.home_team)|(check.team==check.away_team))).all():
        raise ValueError('Box-score season/week/team does not match the schedule; refusing ambiguous timing')
    save(box,'team_game_boxscores.csv'); save(definitions_table(),'stat_definitions.csv')
    print(f'  {len(box)} team-game rows; possession time missing in {int(box.top_seconds.isna().sum())}.')
    avail=None
    if any(f in AVAIL_FAMILIES for f in FEATURE_FAMILIES):
        print('  Player availability: injury reports, weekly rosters, snap counts')
        a=load_availability(years,season)
        targets=pd.concat([schedules[['season','week','home_team']].rename(columns={'home_team':'team'}),
                           schedules[['season','week','away_team']].rename(columns={'away_team':'team'})])
        avail,ainfo=availability_cached(targets,qb,a['snaps'],a['inj'],a['rost'],season,audit=AUDIT)
        print(f'  Availability features: {len(ainfo["cached_seasons"])} completed seasons from cache, '
              f'rebuilt {", ".join(map(str,ainfo["rebuilt_seasons"])) or "none"}')
        save(avail,'availability_features.csv')
        cr=avail[(avail.season==season)&(avail.week==cur)]
        mem=AUDIT[-1]
        share=mem['off_roster_snap_share']
        print(f'  {len(avail)} team-weeks; QB changes projected this week: '
              f'{int((cr.qb_expected.notna()&(cr.qb_expected!=cr.qb_usual)).sum())}')
        print(f'  Roster membership: {"n/a" if share is None else f"{100*share:.1f}%"} of window snaps off the reference roster; '
              f'{mem["membership_skipped_as_data_gap"]} team-weeks skipped as data gaps, {mem["no_reference_roster"]} without a reference roster, '
              f'{mem["reference_older_than_prior_week"]} using an older (post-bye) roster, '
              f'{mem["qb_projected_from_backup_prior"]} QB projections from the backup prior')
    print('[2/4] Building strictly lagged, decayed profiles')
    features={}
    for half in TEAM_HALF_LIVES:
        print(f'  Half-life {half:g} team games')
        features[half]=lagged_features(box,schedules,half,avail)
        save(features[half],f'pregame_features_half_life_{half:g}.csv')
    print('[3/4] Weekly fits and prior-season-only recipe selection')
    oof=cached_grid(features,season)
    save(oof,'candidate_walk_forward_predictions.csv')
    outer,folds=outer_predictions(oof)
    card=scorecard(outer)
    print('Held-out (outer) seasons, pooled:'); display(card.round(4)); save(card,'outer_chronological_scorecard.csv')
    save(outer,'outer_chronological_predictions.csv')
    byyear=pd.concat([scorecard(g).assign(season=int(y)) for y,g in outer.groupby('season')],ignore_index=True)
    if byyear.season.nunique()>1:
        print('Held-out seasons, one at a time:'); display(byyear.round(4))
    save(byyear,'outer_scorecard_by_season.csv')
    blend=market_blend_diagnostic(oof,outer,folds)
    print('Market-blend check (does the composite carry information the spread lacks?):')
    if len(blend): display(blend.round(4))
    print('  '+blend_verdict(blend)); save(blend,'market_blend_diagnostic.csv')
    cal=calibration_bands(outer); save(cal,'calibration_bands.csv')
    dump(outdir/'outer_recipe_choices.json',folds)
    recipe_rec=frozen_recipe(oof,season); recipe=recipe_rec['recipe']
    selection=pd.DataFrame(recipe_rec['selection_scores'])
    print('Recipe selection, best six (full table saved to CSV):')
    display(selection.head(6).round(5)); save(selection,'recipe_selection_diagnostics.csv')
    fam_best=selection.groupby('family')['log loss'].min().sort_values()
    print('Best validation log loss by family (lower is better):'); print(fam_best.round(5).to_string())
    dump(outdir/'frozen_recipe.json',recipe_rec)
    print(f'[4/4] Current recipe: {recipe_key(recipe)}; coefficients refit before each week')
    gw,fit=current_predictions(features,recipe,season,cur)
    save(gw,'current_season_predictions.csv')
    season_card=scorecard(gw); save(season_card,'season_scorecard.csv')
    rcols=['season','week','home won','model_wp','market_wp']
    rec_hist=pd.concat([outer[rcols],gw.loc[(gw.week<cur)&gw.result.notna(),rcols]],ignore_index=True)
    recs=band_records(rec_hist); save(recs,'band_records.csv')
    rec_label=records_source_label(rec_hist,season,cur)
    print(f'Realized pick records in matched bands ({rec_label}):')
    display(band_records_wide(recs))
    board=gw[gw.week==cur].copy()
    save(board,f'week{cur}_board.csv')
    week_card=scorecard(board); save(week_card,'week_scorecard.csv')
    weights=coefficient_table(fit); save(weights,'current_composite_weights.csv')
    save(contribution_table(board,fit),'current_game_score_contributions.csv')
    dump(outdir/'current_coefficient_fit.json',fit)
    html=html_board(board,fit,card,recipe_rec,now.astimezone(ZoneInfo(TIMEZONE)).strftime('%Y-%m-%d %H:%M %Z'),season_card,week_card,calib=None,records=(recs,rec_label))
    (outdir/f'week{cur}_board.html').write_text(html,encoding='utf-8')
    if HAS_IPY: display(HTML(html))
    if WRITE_FORWARD_LEDGER:
        records,new=record_forward(board,fit,recipe_rec)
        graded,forward_card=grade_forward(records,schedules)
        save(graded,'forward_ledger_grades.csv'); save(forward_card,'forward_only_scorecard.csv')
        print(f'First pre-kickoff snapshots added: {new}. Historical reconstructions are excluded from this ledger.')
        if len(forward_card): display(forward_card.round(4))
    sig,cfg=config_signature()
    versions={}
    for name in ('numpy','pandas','scipy','nflreadpy','pyarrow'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='unavailable'
    uses_avail=recipe['family'] in AVAIL_FAMILIES
    uses_adj=profile_family(recipe['family']) in ADJUSTED_FAMILIES
    limitations=['box score definitions approximate some official totals',
                 ('opponent adjustment: additive offense/defense ridge model on prior games; no game-state or schedule-timing adjustment'
                  if uses_adj else 'no explicit opponent-strength adjustment'),
                 'game-state effects in full-game statistics',
                 'design chosen after historical results reviewed; development evaluation, not untouched test',
                 'v1.1-v1.7 changes applied after v1 results were seen',
                 'game-bootstrap intervals omit selection and serial-dependence uncertainty']
    limitations+=(['availability: final injury-report status (no intra-week timestamps); same-week roster status ignored; '
                   'no game-day inactives or late-week news',
                   'roster membership: week 1 uses week-1 roster membership (reserve-listed players there count as out)',
                   'newly acquired QBs have no team history; with no rostered QB history the league backup prior is used']
                  if uses_avail else ['no live starters/injuries in the selected recipe'])
    dump(outdir/'run_manifest.json',{'revision':REVISION,'diagnostics':DIAGNOSTICS,'output_root':str(OUTPUT_ROOT.resolve()),'market_blend_verdict':blend_verdict(blend),
        'generated_utc':now.isoformat(),'season':season,'week':cur,
        'config_signature':sig,'config':cfg,'recipe':recipe,'source_audit':AUDIT,'package_versions':versions,
        'data_mode':'current-upstream historical reconstruction, separate first-pregame local ledger',
        'target':'binary home W/L; ties excluded from fit/evaluation but their prior stats retained',
        'fit_timing':'whole prior weeks only; no within-week coefficient updates',
        'profile_decay':'2^(-team-game age / selected half-life) times offseason retention per season boundary',
        'fit_decay':'2^(-age in NFL-season units / FIT_HALF_LIFE_SEASONS)',
        'validation':'prior-season-only recipe selection; weekly coefficients trained only on earlier game rows',
        'limitations':limitations,
        'csv_files':saved})
    print(f'Saved {len(saved)} CSVs, HTML, fitted weights and audit JSON to {outdir.resolve()}')
    if on_drive:
        print('Frozen recipe, forward ledger and caches are on Google Drive. Let Drive finish syncing '
              '(a minute or so) before disconnecting the runtime.')
    else:
        print('Outputs are LOCAL and are lost when the runtime resets. In Colab, set USE_GOOGLE_DRIVE=True.')
    return outdir


def self_test():
    """Synthetic regressions only. No claim about real NFL predictive accuracy."""
    import tempfile
    rng=np.random.default_rng(SEED)
    # PBP extraction: sacks counted once, net yards, no-play penalties, fumble attribution, TOP dedup.
    s=pd.DataFrame([{'game_id':'toy','season':2020,'week':1,'home_team':'H','away_team':'A','result':3}])
    template={'game_id':'toy','season_type':'REG','two_point_attempt':0,'play_deleted':0,
        'play_type':'pass','posteam':'H','yards_gained':10.,'pass_attempt':1,'rush_attempt':0,
        'sack':0,'interception':0,'fumble_lost':0,'third_down_converted':0,'third_down_failed':0,
        'first_down_rush':0,'first_down_pass':0,'first_down_penalty':0,'penalty':0,'penalty_team':None,
        'penalty_yards':0.,'drive':1,'drive_time_of_possession':'30:00',
        'fumbled_1_team':None,'fumbled_2_team':None,'fumble_recovery_1_team':None,'fumble_recovery_2_team':None}
    plays=[]
    def add(**kw): plays.append({**template,'play_id':len(plays)+1,**kw})
    add(first_down_pass=1,third_down_converted=1)
    add(yards_gained=-7.,sack=1,third_down_failed=1)
    add(play_type='run',pass_attempt=0,rush_attempt=1,yards_gained=4.)
    add(play_type='no_play',pass_attempt=0,yards_gained=0.,penalty=1,penalty_team='A',penalty_yards=15.,first_down_penalty=1)
    add(posteam='A',drive=2,yards_gained=20.,fumble_lost=1,fumbled_1_team='A',fumble_recovery_1_team='H')
    add(posteam='A',drive=2,play_type='run',pass_attempt=0,rush_attempt=1,yards_gained=5.)
    add(posteam='A',drive=2,play_type='no_play',pass_attempt=0,yards_gained=0.,penalty=1,penalty_team='H',penalty_yards=0.)
    b=aggregate_boxscores(pd.DataFrame(plays),s,2020).set_index('team')
    assert b.loc['H','plays']==3 and b.loc['H','sacks_taken']==1
    assert b.loc['H','net_pass_yards']==3 and b.loc['H','net_yards']==7
    assert b.loc['H','pass_plays']==2 and b.loc['H','pass_attempts']==1
    assert b.loc['H','first_downs']==2
    assert b.loc['A','penalties']==1 and b.loc['H','penalties']==0
    assert b.loc['A','fumbles_lost']==1 and b.loc['H','top_seconds']==1800
    # Opponent adjustment: A and B have identical offenses, but A only faced the
    # strong defense S and B only the weak defense W. C faced both, linking them.
    off,dfn,val=[],[],[]
    for o,d,v in [('A','S',4.),('B','W',6.),('C','S',4.),('C','W',6.)]:
        off+=[o]*6; dfn+=[d]*6; val+=[v]*6
    mu,oe,de=opponent_adjust(off,dfn,val,np.ones(len(val)),np.ones(len(val)),.01)
    assert abs((mu+oe['A'])-(mu+oe['B']))<.05 and de['S']<-.9 and de['W']>.9
    # Market blend recovers known weights; a composite equal to the market gets ~zero extra weight.
    brng=np.random.default_rng(1); lm=brng.normal(0,1,6000); lc=brng.normal(0,1,6000)
    yb=(brng.random(6000)<expit(.9*lm+.6*lc)).astype(float)
    bb=fit_blend(expit(lm),expit(lc),yb)
    assert abs(bb[1]-.9)<.15 and abs(bb[2]-.6)<.15
    yb2=(brng.random(6000)<expit(lm)).astype(float)
    assert abs(fit_blend(expit(lm),expit(lc),yb2)[2])<.1
    # Availability: QB swap, unit shares, roster status and membership, bye weeks, no same-game snaps.
    qbt=pd.DataFrame([{'game_id':f'g{w}','season':2020,'week':w,'team':'T','gsis_id':gid,'name':nm,
        'dropbacks':db,'net_yards':db*e,'leader':ld} for w in range(1,5)
        for gid,nm,db,e,ld in (('qa','Starter',36,7.,True),('qb','Backup',3,4.,False))]
        # A receiver with one trick-play throw: never a QB projection.
        +[{'game_id':'g2','season':2020,'week':2,'team':'T','gsis_id':'w1','name':'Gadget','dropbacks':1,'net_yards':30.,'leader':False}]
        +[{'game_id':'o1','season':2019,'week':1,'team':'U','gsis_id':'x','name':'X','dropbacks':40,'net_yards':160.,'leader':False}])
    sn=pd.DataFrame([{'season':2020,'week':w,'game_id':f'g{w}','team':'T','gsis_id':f'p{i}','position':'T',
        'offense_snaps':60.,'defense_snaps':0.} for w in range(1,6) for i in range(5)]
        # p5 played weeks 1-4 and then left the team: no roster row, no status, no injury listing.
        +[{'season':2020,'week':w,'game_id':f'g{w}','team':'T','gsis_id':'p5','position':'T',
           'offense_snaps':60.,'defense_snaps':0.} for w in range(1,5)])
    sn.loc[(sn.week==5)&(sn.gsis_id=='p1'),'offense_snaps']=0.  # same-game snaps must be ignored
    tg=pd.DataFrame({'season':[2020],'week':[5],'team':['T']})
    def roster(active,reserve=('p2',),week=4):
        pos=lambda g:'QB' if g in ('qa','qb') else 'WR' if g=='w1' else 'T'
        return pd.DataFrame([{'season':2020,'week':week,'team':'T','gsis_id':g,'position':pos(g),'status':'ACT'} for g in active]
                            +[{'season':2020,'week':week,'team':'T','gsis_id':g,'position':pos(g),'status':'RES'} for g in reserve]
                            +[{'season':2020,'week':5,'team':'T','gsis_id':'p3','position':'T','status':'RES'}])
    full=('p0','p1','p3','p4','qa','qb','w1')
    rost=roster(full)
    inj=pd.DataFrame([{'season':2020,'week':5,'team':'T','gsis_id':'p0','weight':1.},
                      {'season':2020,'week':5,'team':'T','gsis_id':'p4','weight':.25}])
    aud=[]
    av=availability_table(tg,qbt,sn,inj,rost,audit=aud).iloc[0]
    # p0 out, p2 prior-week RES, p4 questionable, p5 departed; p3 same-week RES ignored.
    assert np.isclose(av.OL_out,(1+1+.25+1)/6) and av.roster_reference_week==4
    assert np.isclose(av.off_roster_snap_share,2/6) and aud[-1]['membership_skipped_as_data_gap']==0
    assert av.qb_expected=='Starter' and abs(av.qb_delta)<.2
    inj2=pd.concat([inj,pd.DataFrame([{'season':2020,'week':5,'team':'T','gsis_id':'qa','weight':1.}])])
    av2=availability_table(tg,qbt,sn,inj2,rost).iloc[0]
    assert av2.qb_expected=='Backup' and av2.qb_delta<-.5
    # A departed starter (not on the reference roster, no injury listing) is not projected.
    av3=availability_table(tg,qbt,sn,inj,roster(('p0','p1','p3','p4','qb','w1'))).iloc[0]
    assert av3.qb_usual=='Starter' and av3.qb_expected=='Backup' and av3.qb_delta<-.5
    # A roster that misses most window players is treated as a data gap, not mass departures.
    aud=[]
    av4=availability_table(tg,qbt,sn,inj,roster((),reserve=('p2',)),audit=aud).iloc[0]
    assert np.isclose(av4.OL_out,(1+1+.25)/6) and aud[-1]['membership_skipped_as_data_gap']==1
    # Post-bye: no week-4 roster row, so the week-3 roster is the reference (status and membership).
    aud=[]
    av5=availability_table(tg,qbt,sn,inj,roster(full,week=3),audit=aud).iloc[0]
    assert av5.roster_reference_week==3 and np.isclose(av5.OL_out,av.OL_out)
    assert aud[-1]['reference_older_than_prior_week']==1 and aud[-1]['no_reference_roster']==0
    # Starter out, backup gone, only a receiver with a trick-play pass left: use the backup prior.
    aud=[]
    av6=availability_table(tg,qbt,sn,inj2,roster(('p0','p1','p3','p4','qa','w1')),audit=aud).iloc[0]
    assert av6.qb_expected==NO_QB_LABEL and av6.qb_delta<-.5 and aud[-1]['qb_projected_from_backup_prior']==1
    # Current-season window: last season's finale stops counting once 2 games this season exist.
    sn2=pd.DataFrame([{'season':2019,'week':17,'game_id':'h17','team':'S','gsis_id':'o1','position':'T','offense_snaps':60.,'defense_snaps':0.}]
        +[{'season':2020,'week':w,'game_id':f'h{w}','team':'S','gsis_id':'n1','position':'T','offense_snaps':60.,'defense_snaps':0.} for w in (1,2)])
    rs2=pd.DataFrame([{'season':2020,'week':2,'team':'S','gsis_id':g,'position':'T','status':'ACT'} for g in ('o1','n1')])
    ij2=pd.DataFrame([{'season':2020,'week':3,'team':'S','gsis_id':'o1','weight':1.}])
    a7=availability_table(pd.DataFrame({'season':[2020],'week':[3],'team':['S']}),qbt,sn2,ij2,rs2).iloc[0]
    assert np.isclose(a7.OL_out,1/3) and np.isclose(a7.OL_out_cs,0.)
    a8=availability_table(pd.DataFrame({'season':[2020],'week':[2],'team':['S']}),qbt,sn2,ij2.assign(week=2),
                          rs2.assign(week=1)).iloc[0]
    assert np.isclose(a8.OL_out_cs,a8.OL_out)  # only 1 current game: falls back to the standard window
    assert 'd__avail__OL_out_cs' in feature_names('rates_core_avail_cs') and 'd__avail__OL_out' not in feature_names('rates_core_avail_cs')
    # v1.7 family: opponent-adjusted core stats plus the current-season availability columns, nothing else.
    fa=feature_names('rates_core_adj_avail_cs')
    assert fa==feature_names('rates_core_adj')+[f'd__avail__{c}' for c in AVAIL_COLS_CS]
    assert not any('__rates_core__' in n for n in fa) and profile_family('rates_core_adj_avail_cs')=='rates_core_adj'
    # Availability cache: identical to a direct build, reused on the 2nd call, rebuilt when an input changes.
    tg2=pd.DataFrame({'season':[2019,2019,2020,2020],'week':[3,4,4,5],'team':['T','T','T','T']})
    old_root=globals()['OUTPUT_ROOT']
    with tempfile.TemporaryDirectory() as tmp:
        globals()['OUTPUT_ROOT']=Path(tmp)
        try:
            direct_aud=[]; direct=availability_table(tg2,qbt,sn,inj,rost,audit=direct_aud)
            for c in ('qb_expected','qb_usual'): direct[c]=direct[c].astype(object).where(direct[c].notna(),None)
            a1,i1=availability_cached(tg2,qbt,sn,inj,rost,2020,audit=(aud1:=[]))
            a2,i2=availability_cached(tg2,qbt,sn,inj,rost,2020,audit=(aud2:=[]))
            key=['season','week','team']
            srt=lambda d:d.sort_values(key).reset_index(drop=True)[direct.columns]
            pd.testing.assert_frame_equal(srt(a1),srt(direct)); pd.testing.assert_frame_equal(srt(a2),srt(direct))
            assert i1['rebuilt_seasons']==[2019,2020] and i2['cached_seasons']==[2019] and i2['rebuilt_seasons']==[2020]
            for k in AVAIL_STAT_KEYS: assert np.isclose(aud2[-1][k],direct_aud[-1][k]),k
            inj3=pd.concat([inj,pd.DataFrame([{'season':2019,'week':4,'team':'T','gsis_id':'p1','weight':1.}])])
            _,i3=availability_cached(tg2,qbt,sn,inj3,rost,2020)
            assert i3['rebuilt_seasons']==[2019,2020] and len(list(Path(tmp,'cache').glob('avail_features_2019_*.pkl')))==1
        finally:
            globals()['OUTPUT_ROOT']=old_root
    assert 'd__avail__qb_delta' in feature_names('rates_core_avail')
    # Relocated-team codes: a 2019 OAK schedule row must match LV play-by-play.
    alias=pd.DataFrame({'home_team':['OAK'],'away_team':['DEN'],'posteam':['LV'],'penalty_team':['SD']})
    alias=normalize_teams(alias)
    assert alias.home_team[0]=='LV' and alias.posteam[0]=='LV' and alias.penalty_team[0]=='LAC'
    # Mirrored possession is excluded from the regression inputs.
    assert 'd__allowed__rates__possession_share_pct' not in feature_names('rates')
    assert 'd__for__rates__possession_share_pct' in feature_names('rates')
    # Synthetic full league with enough repeated team appearances to test lagged profiles.
    schedules=[]; boxes=[]; teams=list('ABCDEFGH')
    for year in range(2018,2025):
        for week in range(1,9):
            perm=rng.permutation(teams)
            for i in range(0,8,2):
                home,away=perm[i],perm[i+1]; gid=f'{year}_{week}_{i}'
                result=int(rng.choice([-14,-7,-3,3,7,14]))
                sp=float(rng.choice([-7,-3,0,3,7]))
                schedules.append({'game_id':gid,'season':year,'week':week,'home_team':home,'away_team':away,
                    'gameday':f'{year}-09-{week+1:02d}','gametime':'13:00','site':float(i!=0),
                    'result':result,'home won':won(result),'spread_line':sp,'market WP':float(ndtr(sp/MARKET_SIGMA))})
                for team,opp in ((home,away),(away,home)):
                    p=float(rng.integers(55,75)); pp=float(rng.integers(27,43)); rush=p-pp
                    py=float(rng.normal(210,50)); ry=float(rng.normal(115,25))
                    boxes.append({'game_id':gid,'season':year,'week':week,'team':team,'opponent':opp,
                        'plays':p,'net_yards':py+ry,'net_pass_yards':py,'rush_yards':ry,'pass_plays':pp,
                        'pass_attempts':pp-2,'rush_attempts':rush,'first_downs':float(rng.integers(12,25)),
                        'third_converted':float(rng.integers(3,9)),'third_attempts':14.,
                        'fumbles_lost':float(rng.integers(0,2)),'interceptions':float(rng.integers(0,3)),
                        'sacks_taken':2.,'penalties':6.,'penalty_yards':45.,'top_seconds':1800.,'game_seconds':3600.})
    box,sched=pd.DataFrame(boxes),pd.DataFrame(schedules)
    keys=['TEAM_HALF_LIVES','RIDGE_GRID','MIN_TRAIN_GAMES','BACKTEST_FIRST_SEASON','OUTER_FIRST_SEASON','OUTPUT_ROOT','BOOTSTRAP_REPS']
    old={k:globals()[k] for k in keys}
    try:
        globals().update(TEAM_HALF_LIVES=(4.,8.),RIDGE_GRID=(.01,.1),MIN_TRAIN_GAMES=20,
                         BACKTEST_FIRST_SEASON=2020,OUTER_FIRST_SEASON=2022,BOOTSTRAP_REPS=100)
        features={h:lagged_features(box,sched,h) for h in TEAM_HALF_LIVES}
        changed=box.copy()
        mask=(changed.season>2023)|((changed.season==2023)&(changed.week>=4))
        changed.loc[mask,'rush_yards']+=1000
        changed['net_yards']=changed.net_pass_yards+changed.rush_yards
        cf=lagged_features(changed,sched,4.)
        beforemask=(features[4.].season<2023)|((features[4.].season==2023)&(features[4.].week<=4))
        names=feature_names('rates')+feature_names('rates_core_adj')[1:]
        np.testing.assert_allclose(features[4.].loc[beforemask,names],cf.loc[beforemask,names],equal_nan=True)
        assert not np.allclose(features[4.].loc[~beforemask,names],cf.loc[~beforemask,names],equal_nan=True)
        # Every production candidate's inputs exist in the feature table.
        for fam in FEATURE_FAMILIES: assert set(feature_names(fam))<=set(features[4.].columns),fam
        # Pure decay arithmetic: observation 8 appearances ago has half weight at h=8.
        assert np.isclose(np.exp2(-8/8),.5)
        oof=walk_forward_grid(features,2023)
        outer,choices=outer_predictions(oof)
        assert (outer.selection_train_through_season<outer.season).all()
        altered=oof.copy(); altered.loc[altered.season>=2022,'home won']=1-altered.loc[altered.season>=2022,'home won']
        altered_outer,altered_choices=outer_predictions(altered)
        first=int(outer.season.min())
        assert choices[0]['recipe']==altered_choices[0]['recipe']
        # Recipe is not allowed to inspect outer labels. Fixed weekly candidate predictions
        # are intentionally unchanged here, so this isolates the selection boundary.
        np.testing.assert_allclose(outer.loc[outer.season==first,'model_wp'],altered_outer.loc[altered_outer.season==first,'model_wp'])
        # Drive migration: copy missing files, keep Drive versions, merge ledger records by key.
        with tempfile.TemporaryDirectory() as tmp:
            loc,dst=Path(tmp)/'local',Path(tmp)/'drive'
            (loc/'cache').mkdir(parents=True); dst.mkdir()
            rec=lambda g:json.dumps({'experiment':'e','game_id':g})
            (loc/'forward_predictions.jsonl').write_text(rec('a')+'\n'+rec('b')+'\n')
            (dst/'forward_predictions.jsonl').write_text(rec('a')+'\n')
            (loc/'frozen_recipe_2024.json').write_text('local'); (dst/'frozen_recipe_2024.json').write_text('drive')
            (loc/'cache'/'x.csv').write_text('1')
            r=migrate_local_outputs(loc,dst)
            assert r=={'copied':1,'kept_existing':1,'ledger_records_added':1}
            assert (dst/'frozen_recipe_2024.json').read_text()=='drive' and (dst/'cache'/'x.csv').read_text()=='1'
            assert [_ledger_key(l)[1] for l in (dst/'forward_predictions.jsonl').read_text().splitlines() if l.strip()]==['a','b']
            assert migrate_local_outputs(loc,dst)['ledger_records_added']==0
        blend=market_blend_diagnostic(oof,outer,choices)
        assert len(blend)>=4 and np.isfinite(blend.estimate).all() and blend_verdict(blend).startswith('Composite weight')
        recipe,table=select_recipe(oof[oof.season<2024])
        # min_log_loss rule must pick the lowest-loss candidate.
        assert table.loc[table.selected,'log loss'].iloc[0]==table['log loss'].min()
        gw,fit=current_predictions(features,recipe,2024,4)
        board=gw[gw.week==4].copy()
        p,z,parts=apply_fit(board,fit,True)
        np.testing.assert_allclose(parts.sum(axis=1),z)
        np.testing.assert_allclose(expit(z),p)
        # No current/future labels or stats may enter coefficient fitting.
        train=features[recipe['half_life']]
        train=train[before(train,2024,4)]
        fit2=fit_composite(train,recipe,2024,4)
        assert fit==fit2
        try: fit_composite(features[recipe['half_life']],recipe,2024,4)
        except ValueError: pass
        else: raise AssertionError('Future training rows accepted')
        # Team swap reverses neutral-site probability, including all offense/defense inputs.
        neutral=board.copy(); neutral['site']=0.
        swapped=neutral.copy()
        for name in feature_names(recipe['family'])[1:]: swapped[name]=-swapped[name]
        np.testing.assert_allclose(apply_fit(neutral,fit)+apply_fit(swapped,fit),1.,atol=1e-12)
        # Freeze + first-snapshot semantics; never overwrite or backfill a final/live game.
        with tempfile.TemporaryDirectory() as tmp:
            globals()['OUTPUT_ROOT']=Path(tmp)
            rec=frozen_recipe(oof,2024)
            assert frozen_recipe(oof.assign(**{'home won':1-oof['home won']}),2024)==rec
            future=board.copy(); future['result']=np.nan; future['gameday']='2030-10-01'; future['gametime']='20:15'
            asof=dt.datetime(2030,10,1,18,tzinfo=dt.timezone.utc)
            records,n=record_forward(future,fit,rec,asof)
            assert n==len(future)
            first_bytes=(Path(tmp)/'forward_predictions.jsonl').read_bytes()
            _,n=record_forward(future,fit,rec,asof+dt.timedelta(minutes=1))
            assert n==0 and (Path(tmp)/'forward_predictions.jsonl').read_bytes()==first_bytes
            late=future.copy(); late['game_id']=late.game_id+'late'
            _,n=record_forward(late,fit,rec,asof+dt.timedelta(days=1)); assert n==0
            final=future.copy(); final['game_id']=final.game_id+'final'; final['result']=3.
            _,n=record_forward(final,fit,rec,asof); assert n==0
            _,forward=grade_forward(records,sched)
            assert len(forward)==3
            mixed=pd.concat([board,future.assign(game_id=future.game_id+'up')],ignore_index=True)
            cal=calibration_bands(outer)
            assert cal.groupby('source').games.sum().eq(len(outer)).all() and len(cal)==2*(len(CAL_BAND_EDGES)-1)
            lo,hi=wilson(5,10); assert abs(lo-.237)<.002 and abs(hi-.763)<.002
            html=html_board(mixed,fit,scorecard(outer),rec,'synthetic test',scorecard(gw),scorecard(board),calib=cal)
            assert 'Calibration in matched bands' in html and html.count('<circle')==int((cal.games>0).sum())
            rh=pd.concat([outer[['season','week','home won','model_wp','market_wp']],
                          gw.loc[gw.week<4,['season','week','home won','model_wp','market_wp']]],ignore_index=True)
            recs=band_records(rh); tot=recs[recs.band=='All picks'].set_index('source')
            bands=recs[recs.band!='All picks'].groupby('source')[['wins','games']].sum()
            assert (bands.wins.sort_index()==tot.wins.sort_index()).all() and (bands.games.sort_index()==tot.games.sort_index()).all()
            yy=rh['home won']; pp=rh.model_wp
            assert tot.loc['model','wins']==int((((pp>.5)&(yy==1))|((pp<.5)&(yy==0))).sum())
            assert record_text(3,1)=='3-1 (.750)' and pick_band_index(.31)==3 and pick_band_index(.5001)==0 and pick_band_index(.95)==6
            html2=html_board(mixed,fit,scorecard(outer),rec,'synthetic test',scorecard(gw),scorecard(board),
                             records=(recs,records_source_label(rh,2024,4)))
            assert 'Realized records in matched bands' in html2 and html2.count('slate recs')==len(mixed)
            assert 'Composite log-odds edge' in html and 'Largest score contributions' in html
            assert 'game-state effects' in html and 'Upcoming' in html and 'Final:' in html
            assert 'Availability' in html or recipe['family'] not in AVAIL_FAMILIES
            assert html.count('<article')==len(mixed)
            # Gap warning and footer must match the recipe: availability claims and opponent adjustment.
            gapped=mixed.copy(); gapped['market_wp']=np.where(gapped.model_wp>.5,.01,.99)
            for fam,says,never,adj in (('rates_core_avail','final injury reports','no injury, quarterback',False),
                                       ('rates_core','no injury, quarterback','final injury reports',False),
                                       ('rates_core_adj_avail_cs','final injury reports','no injury, quarterback',True)):
                h=html_board(gapped,fit,scorecard(outer),dict(rec,recipe=dict(rec['recipe'],family=fam)),'t')
                assert says in h and never not in h
                assert ('Profiles are opponent-adjusted.' in h)==adj and ('No explicit opponent-strength adjustment.' in h)==(not adj)
            (Path(tmp)/'board.html').write_text(html,encoding='utf-8')
            Path('selftest_board.html').write_text(html,encoding='utf-8')
        print('PASS: box-score definitions, same-game/future-stat exclusion, roster membership, departed players, bye weeks, '
              'QB-position projection, current-season window, opponent-adjusted availability family, market blend, '
              'matched-band calibration and pick records, Drive migration, availability cache, '
              'decay, chronological selection, weekly fit boundary, exact score decomposition, team symmetry, '
              'frozen recipe, first-pregame ledger and board rendering.')
        return box,sched
    finally:
        globals().update(old)


if __name__=='__main__':
    if '--self-test' in sys.argv: self_test()
    else: main()

