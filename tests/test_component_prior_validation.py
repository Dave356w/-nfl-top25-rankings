import unittest
import pandas as pd
from tools.build_component_priors import build,prepare


def weekly():
    rows=[]
    for year in (2023,2024,2025):
        for week in range(1,18):
            rows.append(dict(player_id='p',player_display_name='Example',position='WR',
                season_type='REG',season=year,week=week,passing_yards=0,passing_tds=0,
                passing_interceptions=0,rushing_yards=0,rushing_tds=0,receiving_yards=week*3,
                receiving_tds=int(week%4==0),receptions=week%5,
                sack_fumbles_lost=0,rushing_fumbles_lost=0,receiving_fumbles_lost=0))
    return pd.DataFrame(rows)


class ComponentPriorValidationTests(unittest.TestCase):
    def test_holdout_cannot_select_recency_parameter(self):
        data=weekly()
        first=build(data,2025,'2026-09-10T00:00:00Z')
        data.loc[data.season.eq(2025),'receiving_yards']=10000
        second=build(data,2025,'2026-09-10T00:00:00Z')
        self.assertEqual(first['validation']['WR']['alpha'],second['validation']['WR']['alpha'])
        self.assertFalse(first['validation']['WR']['approved'])
        self.assertEqual(first['priors'],[])

    def test_future_season_and_duplicate_records_are_rejected(self):
        data=weekly()
        with self.assertRaisesRegex(ValueError,'ended'):
            build(data,2025,'2025-09-01T00:00:00Z')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            prepare(pd.concat([data,data.iloc[:1]]))

    def test_unknown_component_is_not_zero_filled(self):
        data=weekly();data.loc[0,'passing_tds']=float('nan')
        self.assertEqual(len(prepare(data)),len(data)-1)

