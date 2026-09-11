import unittest

import pandas as pd

from pipeline import team_outcome as to


TEAMS=('STL','SEA','GB','CHI')


def schedules(unplayed_week=None):
    """Two small seasons: one tie, one neutral site, one bye, one future game."""
    rows=[]
    for season in (2019,2020):
        for week in range(1,6):
            pairs=[(TEAMS[0],TEAMS[1]),(TEAMS[2],TEAMS[3])] if week%2 else [(TEAMS[1],TEAMS[2]),(TEAMS[3],TEAMS[0])]
            if season==2020 and week==3: pairs=[(TEAMS[0],TEAMS[1])]  # GB and CHI on a bye
            for home,away in pairs:
                played=not (season==2020 and week==unplayed_week)
                # Deterministic and never accidentally tied: exactly one tie is
                # scripted, so the tie assertions cannot drift with a seed.
                base=17.0+(week*3)%11+(season-2019)*2
                tie=season==2019 and week==3 and home==TEAMS[0]
                home_score=base if played else None
                away_score=(base if tie else base-3-(week%5)-(0 if home==TEAMS[0] else 4)) if played else None
                rows.append(dict(game_id=f'{season}_{week:02d}_{away}_{home}',season=season,week=week,
                    game_type='REG',gameday=f'{season}-09-{week:02d}',gametime='13:00',
                    home_team=home,away_team=away,
                    location='Neutral' if (season==2019 and week==4 and home==TEAMS[2]) else 'Home',
                    roof='dome' if home==TEAMS[0] else 'outdoors',surface='grass',
                    div_game=int(week%2==0),home_rest=4 if week==3 else 7,away_rest=7,
                    stadium_id=f'ST{home}',home_score=home_score,away_score=away_score,
                    home_moneyline=-150.0,away_moneyline=130.0,spread_line=-3.0,
                    home_spread_odds=-110.0,away_spread_odds=-110.0,total_line=44.0,
                    over_odds=-110.0,under_odds=-110.0,
                    result=(home_score-away_score) if played else None,
                    total=(home_score+away_score) if played else None,overtime=0,
                    temp=70.0,wind=5.0,referee='R',home_qb_id='q1',away_qb_id='q2',
                    home_qb_name='A',away_qb_name='B',home_coach='C',away_coach='D'))
    # A postseason week so the ordering guard has something to check.
    rows.append(dict(rows[-1],game_id='2020_06_GB_SEA',season=2020,week=6,game_type='WC',
        home_team='SEA',away_team='GB',home_score=24.0,away_score=20.0,result=4.0,total=44.0))
    return pd.DataFrame(rows)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.corpus=to.build_corpus(schedules())

    def test_context_frame_carries_no_outcome_or_market_column(self):
        withheld=set(to.MARKET_COLUMNS)|set(to.OUTCOME_SOURCE_COLUMNS)|set(to.WITHHELD_COLUMNS)
        self.assertFalse(withheld&set(self.corpus.games.columns))
        # and the frames still describe the same games
        self.assertEqual(len(self.corpus.games),len(schedules()))
        self.assertTrue(set(self.corpus.outcomes.game_id)<=set(self.corpus.games.game_id))

    def test_history_is_strictly_earlier(self):
        for season,week in self.corpus.weeks():
            history=self.corpus.history(season,week)
            if history.empty: continue
            order=history.season*100+history.week
            self.assertLess(int(order.max()),season*100+week)

    def test_tie_is_half_a_win_and_never_dropped(self):
        ties=self.corpus.outcomes[self.corpus.outcomes.tie]
        self.assertEqual(len(ties),1)
        self.assertEqual(float(ties.home_win.iloc[0]),0.5)
        self.assertEqual(float(ties.margin.iloc[0]),0.0)

    def test_relocated_franchise_keeps_one_history(self):
        self.assertIn('LAR',set(self.corpus.games.home_team)|set(self.corpus.games.away_team))
        self.assertNotIn('STL',set(self.corpus.games.home_team)|set(self.corpus.games.away_team))

    def test_unplayed_game_is_scheduled_but_has_no_outcome(self):
        corpus=to.build_corpus(schedules(unplayed_week=5))
        future=corpus.games[corpus.games.season.eq(2020)&corpus.games.week.eq(5)]
        self.assertTrue(len(future))
        self.assertFalse(set(future.game_id)&set(corpus.outcomes.game_id))
        # a scheduled game still gets features, which is the live use case
        self.assertEqual(len(to.feature_frame(corpus,2020,5)),len(future))

    def test_market_frame_is_separate_and_joinable(self):
        self.assertEqual(set(self.corpus.market.columns)-{'game_id'},set(to.MARKET_COLUMNS))
        self.assertTrue(set(self.corpus.market.game_id)<=set(self.corpus.games.game_id))

    def test_malformed_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate game_id'):
            to.build_corpus(pd.concat([schedules(),schedules().iloc[:1]]))
        partial=schedules();partial.loc[0,'away_score']=None
        with self.assertRaisesRegex(ValueError,'one score'):
            to.build_corpus(partial)
        overlapping=schedules();overlapping.loc[overlapping.index[-1],'week']=5
        with self.assertRaisesRegex(ValueError,'postseason week numbering'):
            to.build_corpus(overlapping)
        with self.assertRaisesRegex(ValueError,'missing required columns'):
            to.build_corpus(schedules().drop(columns=['home_rest']))


class AsOfTests(unittest.TestCase):
    def setUp(self):
        self.corpus=to.build_corpus(schedules())

    def test_features_do_not_move_when_the_future_is_replaced(self):
        report=to.audit(self.corpus)
        self.assertTrue(report['clean'],report['findings'])
        self.assertEqual(report['checkpoints'],len(self.corpus.weeks(settled_only=True)))

    def test_audit_detects_an_off_by_one_as_of_boundary(self):
        class LeakyCorpus(to.Corpus):
            """The regression this harness exists to prevent: <= instead of <."""
            def history(self,season,week):
                joined=self.games.merge(self.outcomes,on='game_id',how='inner')
                order=joined.season*100+joined.week
                return joined.loc[order.le(season*100+week)].reset_index(drop=True)

        leaky=LeakyCorpus(games=self.corpus.games,outcomes=self.corpus.outcomes,
                          market=self.corpus.market)
        report=to.audit(leaky)
        self.assertFalse(report['clean'])
        checks={f['check'] for f in report['findings']}
        self.assertIn('history_boundary',checks)
        self.assertIn('perturbation',checks)
        moved={f['feature'] for f in report['findings'] if f['check']=='perturbation'}
        # Every feature that reads history is served the corrupted week and moves;
        # every context-only feature is untouched, which is what pins the finding
        # to the boundary rather than to the features.
        self.assertEqual(moved,{'prior_margin_diff','elo_diff'})
        self.assertFalse(moved&{'home_field','dome','div_game','neutral_site','rest_diff'})

    def test_audit_detects_a_withheld_column_reaching_the_view(self):
        games=self.corpus.games.copy()
        games['spread_line']=-3.0
        corpus=to.Corpus(games=games,outcomes=self.corpus.outcomes,market=self.corpus.market)
        report=to.audit(corpus,checkpoints=[(2020,4)])
        self.assertFalse(report['clean'])
        self.assertTrue(any(f['check']=='withheld_columns' for f in report['findings']))

    def test_rolling_feature_uses_only_earlier_games(self):
        # week 1 of the first season has no history at all
        first=to.feature_frame(self.corpus,2019,1)
        self.assertTrue((first.prior_margin_diff==0).all())
        later=to.feature_frame(self.corpus,2020,5)
        self.assertTrue((later.prior_margin_diff!=0).any())

    def test_bye_comes_from_the_published_schedule(self):
        frame=to.feature_frame(self.corpus,2020,4).set_index('game_id')
        # GB and CHI sat out 2020 week 3; both play in week 4, one home one away
        row=frame[frame.index.str.contains('GB')|frame.index.str.contains('CHI')]
        self.assertTrue((row.bye_diff.abs()>0).any())
        self.assertTrue((to.feature_frame(self.corpus,2019,1).bye_diff==0).all())

    def test_design_matrix_carries_targets_but_no_target_shaped_feature(self):
        design=to.build_design(self.corpus)
        self.assertEqual(len(design),len(self.corpus.outcomes))
        self.assertEqual(set(design.columns)-set(to.FEATURES)-{'game_id','season','week'},
                         {'margin','home_win','tie'})
        self.assertFalse({'home_score','away_score'}&set(design.columns))


class EloTests(unittest.TestCase):
    def setUp(self):
        self.corpus=to.build_corpus(schedules())

    def test_winning_raises_a_rating_and_the_loser_pays_for_it(self):
        history=self.corpus.history(2019,5)
        ratings=to.elo_ratings(history)
        self.assertAlmostEqual(sum(ratings.values()),to.ELO_START*len(ratings),places=6)
        won=history.groupby('home_team').margin.sum()
        best=won.idxmax()
        self.assertGreater(ratings[best],to.ELO_START)

    def test_ratings_regress_between_seasons(self):
        history=self.corpus.history(2020,1)
        end_of_2019=to.elo_ratings(history)
        into_2020=to.elo_ratings(history,target_season=2020)
        for team,rating in end_of_2019.items():
            self.assertLess(abs(into_2020[team]-to.ELO_START),abs(rating-to.ELO_START)+1e-9)

    def test_elo_is_covered_by_the_as_of_audit(self):
        self.assertIn('elo_diff',to.FEATURES)
        report=to.audit(self.corpus,features={'elo_diff':to.FEATURES['elo_diff']})
        self.assertTrue(report['clean'],report['findings'])


class SealTests(unittest.TestCase):
    def test_sealed_seasons_are_dropped_from_summaries(self):
        frame=pd.DataFrame({'season':[2019,2024,2025],'x':[1,2,3]})
        self.assertEqual(list(to.unsealed(frame).season),[2019])

    def test_evaluation_seasons_exclude_the_seal_and_the_training_floor(self):
        rows=[]
        for season in range(2010,2026):
            rows.append(dict(game_id=f'{season}_x',season=season,week=1,game_type='REG',
                gameday=f'{season}-09-01',gametime='13:00',home_team='GB',away_team='CHI',
                location='Home',roof='dome',surface='grass',div_game=1,home_rest=7,
                away_rest=7,stadium_id='S',home_score=20.0,away_score=17.0))
        corpus=to.build_corpus(pd.DataFrame(rows))
        seasons=to.evaluation_seasons(corpus)
        self.assertFalse(set(seasons)&set(to.SEALED_SEASONS))
        self.assertEqual(min(seasons),2010+to.MINIMUM_TRAINING_SEASONS)
        self.assertEqual(max(seasons),2023)


if __name__=='__main__':
    unittest.main()
