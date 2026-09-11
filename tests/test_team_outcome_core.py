import unittest

import pandas as pd

from pipeline import team_outcome as to
from pipeline import team_outcome_core as core


TEAMS=('STL','SEA','GB','CHI')


def schedules():
    rows=[]
    for season in (2019,2020):
        for week in range(1,6):
            pairs=[(TEAMS[0],TEAMS[1]),(TEAMS[2],TEAMS[3])] if week%2 else [(TEAMS[1],TEAMS[2]),(TEAMS[3],TEAMS[0])]
            for home,away in pairs:
                base=17.0+(week*3)%11
                rows.append(dict(game_id=f'{season}_{week:02d}_{away}_{home}',season=season,week=week,
                    game_type='REG',gameday=f'{season}-09-{week:02d}',gametime='13:00',
                    home_team=home,away_team=away,location='Home',roof='dome',surface='grass',
                    div_game=0,home_rest=7,away_rest=7,stadium_id='S',
                    home_score=base,away_score=base-7))
    return pd.DataFrame(rows)


def fantasy(strong='GB'):
    """Every team fields a full core; one team's units score far more."""
    rows=[]
    for season in (2019,2020):
        for week in range(1,6):
            for team in TEAMS:
                alias=to._team(team)
                for position,count in core.FIXED_CORE:
                    # One spare beyond each slot, so the core has to choose. A
                    # defence is a single unit per team, so it gets no spare.
                    spares=1 if position!='DEF' else 0
                    for slot in range(count+spares):
                        points=(20.0 if alias==to._team(strong) else 5.0)-slot
                        rows.append(dict(season=season,week=week,team=team,
                                         unit_id=f'{team}:{position}:{slot}',
                                         position=position,points=points))
    return core.build_fantasy(
        pd.DataFrame([r for r in rows if r['position']!='DEF']).rename(columns={'unit_id':'player_id'}),
        pd.DataFrame([r for r in rows if r['position']=='DEF']))


class ScoringTests(unittest.TestCase):
    def test_points_allowed_tiers_match_the_published_table(self):
        bonus=core.points_allowed_bonus([0,3,6,7,13,14,20,21,27,28,34,35,60])
        self.assertEqual(list(bonus),[10,7,7,4,4,1,1,0,0,-1,-1,-4,-4])

    def test_defensive_points_add_events_to_the_shutout_bonus(self):
        frame=pd.DataFrame([{'def_sacks':3.0,'def_interceptions':2.0,'def_fumbles':1.0,
                             'def_tds':1.0,'def_safeties':0.0,'def_fg_blocks':0.0,
                             'def_pat_blocks':0.0,'def_punt_blocks':0.0,'special_teams_tds':0.0}])
        # 3 sacks + 2 INT + 1 fumble + 1 TD = 3 + 4 + 2 + 6 = 15, plus a shutout's 10
        self.assertEqual(float(core.defensive_points(frame,pd.Series([0.0])).iloc[0]),25.0)

    def test_a_duplicate_unit_week_is_refused(self):
        good=fantasy()
        with self.assertRaisesRegex(ValueError,'Duplicate unit-weeks'):
            core.build_fantasy_frame(good)  # fine
            core.build_fantasy(pd.concat([good,good]).rename(columns={'unit_id':'player_id'}),None)


class CoreConstructionTests(unittest.TestCase):
    def setUp(self):
        self.corpus=core.attach_fantasy(to.build_corpus(schedules()),fantasy())

    def test_expectations_lag_and_follow_the_last_team_played_for(self):
        history=self.corpus.fantasy_before(2020,3)
        frame=core.expectations(history)
        self.assertTrue((frame.appearances<=core.FORM_WINDOW).all())
        self.assertEqual(set(frame.position),set(core.CORE_POSITIONS))
        self.assertTrue(frame.team.notna().all())

    def test_the_core_takes_exactly_the_specified_slots(self):
        cores=core.team_cores(self.corpus.fantasy_before(2020,3))
        self.assertEqual(sorted(cores.team),sorted({to._team(t) for t in TEAMS}))
        self.assertTrue((cores.slots_filled==core.CORE_SIZE).all())

    def test_the_core_prefers_the_higher_expectation_within_a_slot(self):
        history=self.corpus.fantasy_before(2020,3)
        cores=core.team_cores(history).set_index('team')
        # the strong team's ten best must outscore a weak team's ten best
        self.assertGreater(float(cores.loc['GB','core']),float(cores.loc['CHI','core']))

    def test_the_edge_is_the_home_core_minus_the_away_core(self):
        view=self.corpus.view(2020,3)
        edge=core.core_edge(view)
        cores=core.team_cores(view.fantasy).set_index('team').core
        for position,row in enumerate(view.context.itertuples(index=False)):
            expected=float(cores[row.home_team])-float(cores[row.away_team])
            self.assertAlmostEqual(float(edge.edge.iloc[position]),expected,places=6)
        self.assertTrue(edge.complete.all())

    def test_tiers_split_on_the_neutral_band(self):
        labelled=core.tiers(pd.Series([-5.0,-0.5,0.0,0.5,5.0]),neutral=1.0)
        self.assertEqual(list(labelled),['negative','neutral','neutral','neutral','positive'])


class CoreLeakageTests(unittest.TestCase):
    def setUp(self):
        self.corpus=core.attach_fantasy(to.build_corpus(schedules()),fantasy())
        self.features={n:to.FEATURES[n] for n in ('fixed_core_diff','fixed_core_complete')}

    def test_the_view_only_carries_earlier_unit_weeks(self):
        for season,week in self.corpus.weeks():
            frame=self.corpus.view(season,week).fantasy
            if frame.empty: continue
            self.assertLess(int((frame.season*100+frame.week).max()),season*100+week)

    def test_the_core_passes_the_as_of_audit(self):
        report=to.audit(self.corpus,features=self.features)
        self.assertTrue(report['clean'],report['findings'])

    def test_a_broken_boundary_is_caught_on_the_fantasy_frame(self):
        # The lesson from the efficiency frame: a new settled-data path needs its
        # own canary, or the gate silently covers one boundary and not the other.
        self.assertTrue(to.canary_detected(self.corpus,features=self.features))
        broken=to._OffByOneCorpus(games=self.corpus.games,outcomes=self.corpus.outcomes,
                                  market=self.corpus.market,efficiency=self.corpus.efficiency,
                                  fantasy=self.corpus.fantasy)
        checks={f['check'] for f in to.audit(broken,features=self.features)['findings']}
        self.assertIn('fantasy_boundary',checks)

    def test_a_corpus_without_fantasy_still_builds_the_feature(self):
        bare=to.build_corpus(schedules())
        frame=to.feature_frame(bare,2020,3,features=self.features)
        self.assertTrue((frame.fixed_core_diff==0).all())
        self.assertTrue((frame.fixed_core_complete==0).all())


if __name__=='__main__':
    unittest.main()
