from types import SimpleNamespace

import pandas as pd

from pipeline import portfolio_construction as pc


def _cfg(entries=4):
    return SimpleNamespace(
        lineup_size=5,
        tournament_lineups=entries,
        max_player_exposure=1.0,
        max_superstar_exposure=1.0,
        max_shared_players=4,
    )


def test_default_construction_mix_is_twenty_entries():
    rules = pc.serializable_rules()
    assert sum(rule["count"] for rule in rules) == 20
    assert pc.apportioned_rule_counts(rules, 20) == [4, 3, 4, 3, 2, 2, 2]
    for rule in rules:
        assert sum(rule["positions"].values()) == 5


def test_rule_counts_scale_to_requested_portfolio_size():
    rules = pc.serializable_rules()
    counts = pc.apportioned_rule_counts(rules, 10)
    assert sum(counts) == 10
    assert all(count >= 0 for count in counts)
    assert len(pc.rule_schedule(rules, 10)) == 10


def test_selector_assigns_different_rules_per_entry():
    players = pd.DataFrame({
        "Position": ["QB", "QB", "WR", "WR", "RB", "TE", "DEF", "WR", "RB", "WR"],
    })
    rules = [
        {
            "name": "2QB / 2WR / 1RB",
            "count": 1,
            "positions": {"QB": 2, "RB": 1, "WR": 2, "TE": 0, "DEF": 0},
        },
        {
            "name": "1QB / 2WR / 1RB / 1TE",
            "count": 1,
            "positions": {"QB": 1, "RB": 1, "WR": 2, "TE": 1, "DEF": 0},
        },
    ]
    scored = pd.DataFrame([
        {"Player_Ids": (0, 1, 2, 3, 4), "Superstar_Id": 0, "Tournament_Score": 10.0},
        {"Player_Ids": (0, 2, 3, 4, 5), "Superstar_Id": 0, "Tournament_Score": 9.0},
        {"Player_Ids": (0, 1, 2, 7, 8), "Superstar_Id": 1, "Tournament_Score": 8.0},
        {"Player_Ids": (1, 2, 7, 8, 5), "Superstar_Id": 1, "Tournament_Score": 7.0},
    ])

    portfolio = pc.select_portfolio(scored, players, _cfg(4), rules)

    assert len(portfolio) == 4
    assert portfolio["Construction_Rule"].tolist() == [
        "2QB / 2WR / 1RB",
        "1QB / 2WR / 1RB / 1TE",
        "2QB / 2WR / 1RB",
        "1QB / 2WR / 1RB / 1TE",
    ]
    for _, row in portfolio.iterrows():
        rule = next(rule for rule in rules if rule["name"] == row["Construction_Rule"])
        assert pc.lineup_matches_rule(players, row["Player_Ids"], rule)


def test_selector_reports_unfilled_rule_slots_instead_of_breaking_shape():
    players = pd.DataFrame({"Position": ["QB", "WR", "WR", "RB", "TE"]})
    scored = pd.DataFrame([
        {"Player_Ids": (0, 1, 2, 3, 4), "Superstar_Id": 0, "Tournament_Score": 1.0},
    ])
    rules = [{
        "name": "2QB required",
        "count": 1,
        "positions": {"QB": 2},
    }]

    portfolio = pc.select_portfolio(scored, players, _cfg(1), rules)

    assert portfolio.empty
    assert portfolio.attrs["construction_unfilled"] == {"2QB required": 1}
