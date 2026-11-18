"""Market-implied fantasy points from Bovada and Underdog player props.

Restored from the pre-e4a6d21 notebook (cell 5), unchanged, so the parsing,
vig removal and component scoring match the week-1 market projections in the
projection archive. Used for display only: the 2026 week-1 backtest found
adding it to the Sleeper+ESPN+CBS mean did not lower error (+0.018 MAE, 95% CI
-0.035 to +0.071, n=142), so it never feeds the optimizer.

Original engine docstring follows.
"""
"""Market-implied NFL fantasy projections from Underdog and Bovada props.

Why this version differs from the original:

* Alternate lines are survival probabilities (P[stat >= threshold]).  They
  cannot be normalized and averaged as if they were probability masses.
* Underdog supplies broad, paired main lines for the full slate. Bovada adds
  alternate ladders that help estimate the shape of each stat distribution.
* Two-way main props from either feed are de-vigged and used as anchors.
* One-way alternate prices are adjusted in log-odds space.  The adjustment is
  learned from main/alternate pairs on the same slate when possible.
* Yardage is fit with a non-negative Weibull survival curve.  Receptions and
  touchdown counts are fit with a Poisson survival curve.
* Players are merged across feeds only when normalized name, team, and game
  time agree; different games can never be silently combined.
* Passing interceptions and lost fumbles are included when markets exist.
* Touchdown-only players receive a slate-trained fantasy-point estimate.  The
  regression uses only good, non-quarterback projections with touchdown props
  as its targets and is labeled separately from component-based projections.

This is a market-derived estimate, not a predictive guarantee. Both feeds are
public but undocumented and may change. Make one request to each feed per run
and cache the payloads with the input flags if you are iterating.

The offensive projection omits components without a dependable prop market,
including two-point conversions and return statistics. It does not project
kickers or D/ST.
"""


import argparse
import base64
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BOVADA_ENDPOINT = (
    "https://www.bovada.lv/services/sports/event/v2/events/A/"
    "description/football/nfl"
)
UNDERDOG_ENDPOINTS = (
    "https://api.underdogfantasy.com/v2/over_under_lines"
    "?product=fantasy&sport_id=NFL",
    "https://api.underdogfantasy.com/beta/v6/over_under_lines",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
)
# Backwards-compatible singular name now points to the current endpoint.
UNDERDOG_ENDPOINT = UNDERDOG_ENDPOINTS[0]

# Backwards-compatible name for notebooks that imported ENDPOINT.
ENDPOINT = BOVADA_ENDPOINT

USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/15.0 Mobile/15E148 Safari/604.1"
)

UNDERDOG_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

EPSILON = 1e-6
DEFAULT_FALLBACK_LOGIT_VIG = 0.17
MIN_GLOBAL_TD_REGRESSION_SAMPLES = 12
MIN_POSITION_TD_REGRESSION_SAMPLES = 8
TD_REGRESSION_RESIDUAL_MAD_LIMIT = 3.5

YARD_STATS = {"passing_yards", "rushing_yards", "receiving_yards"}
COUNT_STATS = {
    "receptions",
    "passing_touchdowns",
    "any_touchdowns",
    "interceptions",
    "fumbles_lost",
}

STAT_ALIASES = {
    "Passing Yards": "passing_yards",
    "Rushing Yards": "rushing_yards",
    "Receiving Yards": "receiving_yards",
    "Receptions": "receptions",
    "Passing Touchdowns": "passing_touchdowns",
    "Interceptions Thrown": "interceptions",
    "Passing Interceptions": "interceptions",
}

DEFAULT_WEIBULL_SHAPES = {
    "passing_yards": 4.0,
    "rushing_yards": 2.0,
    "receiving_yards": 2.0,
}

STAT_LABELS = {
    "passing_yards": "PaY",
    "rushing_yards": "RuY",
    "receiving_yards": "ReY",
    "receptions": "Rec",
    "passing_touchdowns": "PaTD",
    "any_touchdowns": "TD",
    "interceptions": "INT",
    "fumbles_lost": "FUM",
}

UNDERDOG_STAT_ALIASES = {
    "passing_yds": "passing_yards",
    "rushing_yds": "rushing_yards",
    "receiving_yds": "receiving_yards",
    "receiving_rec": "receptions",
    "passing_tds": "passing_touchdowns",
    "passing_ints": "interceptions",
    "rush_rec_tds": "any_touchdowns",
    "fumbles_lost": "fumbles_lost",
}

TEAM_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "NOR": "NO",
    "WSH": "WAS",
}

NFL_TEAM_NAMES = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LAR",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}


@dataclass(frozen=True)
class Scoring:
    passing_yard: float
    rushing_yard: float
    receiving_yard: float
    reception: float
    passing_touchdown: float
    rushing_receiving_touchdown: float
    interception: float
    fumble_lost: float


SCORING_PRESETS = {
    # Current Yahoo default offensive categories: half-PPR, 1/25 passing,
    # 1/10 rushing/receiving, 4-point passing TD, -1 per interception, and
    # -2 per lost fumble.
    "yahoo": Scoring(0.04, 0.10, 0.10, 0.50, 4.0, 6.0, -1.0, -2.0),
    "half-ppr": Scoring(0.04, 0.10, 0.10, 0.50, 4.0, 6.0, -2.0, -2.0),
    "ppr": Scoring(0.04, 0.10, 0.10, 1.00, 4.0, 6.0, -2.0, -2.0),
    # Preserves the coefficients and behavior of the supplied script.  Its
    # declared interceptions variable was unused, so the penalty remains 0.
    "original": Scoring(0.06, 0.125, 0.125, 1.00, 4.0, 6.0, 0.0, 0.0),
}


@dataclass
class Observation:
    threshold: float
    probability: float
    weight: float
    source: str


@dataclass
class StatMarket:
    alternate: Dict[float, List[float]] = field(default_factory=dict)
    anchors: List[Observation] = field(default_factory=list)
    market_ids: set[str] = field(default_factory=set)
    providers: set[str] = field(default_factory=set)

    def add_alternate(
        self,
        threshold: float,
        probability: float,
        market_id: str,
        provider: str = "bovada",
    ) -> None:
        self.alternate.setdefault(float(threshold), []).append(probability)
        self.providers.add(provider)
        if market_id:
            self.market_ids.add(f"{provider}:{market_id}")

    def add_anchor(
        self,
        threshold: float,
        probability: float,
        market_id: str,
        provider: str = "bovada",
    ) -> None:
        self.anchors.append(
            Observation(float(threshold), probability, 4.0, f"{provider} total")
        )
        self.providers.add(provider)
        if market_id:
            self.market_ids.add(f"{provider}:{market_id}")


@dataclass
class PlayerMarkets:
    event_id: str
    name: str
    team: str
    matchup: str
    start_time_ms: Optional[int]
    position: str = "UNK"
    stats: Dict[str, StatMarket] = field(default_factory=dict)

    def market(self, stat: str) -> StatMarket:
        return self.stats.setdefault(stat, StatMarket())


@dataclass
class Distribution:
    family: str
    mean: float
    parameter_1: float
    parameter_2: Optional[float] = None

    def survival(self, threshold: float) -> float:
        if self.family == "weibull":
            shape = self.parameter_1
            scale = self.parameter_2 or EPSILON
            if threshold <= 0:
                return 1.0
            return math.exp(-((threshold / scale) ** shape))
        if self.family == "poisson":
            return poisson_survival(max(1, int(math.ceil(threshold))), self.parameter_1)
        raise ValueError(f"Unknown distribution family: {self.family}")


@dataclass
class Projection:
    event_id: str
    matchup: str
    start_time_utc: Optional[str]
    team: str
    player: str
    position: str
    fantasy_points: float
    quality: str
    stat_means: Dict[str, float]
    sources: Dict[str, str]
    fantasy_points_method: str = "component-sum"


@dataclass(frozen=True)
class TdRegression:
    """Linear expected-TD to fantasy-point fit learned from this slate."""

    label: str
    intercept: float
    slope: float
    sample_count: int
    r_squared: float
    maximum_target: float

    def predict(self, expected_touchdowns: float, touchdown_points: float) -> float:
        estimate = self.intercept + self.slope * max(0.0, expected_touchdowns)
        # A complete estimate should not fall below its known touchdown scoring
        # component.  Cap extreme extrapolation just beyond the strongest good
        # projection observed on the same slate.
        floor = max(0.0, expected_touchdowns * touchdown_points)
        ceiling = max(floor, self.maximum_target * 1.10)
        return min(ceiling, max(floor, estimate))


def clamp_probability(value: float) -> float:
    return min(1.0 - EPSILON, max(EPSILON, float(value)))


def logit(probability: float) -> float:
    p = clamp_probability(probability)
    return math.log(p / (1.0 - p))


def logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def american_odds_to_probability(american_odds: Any) -> Optional[float]:
    """Convert American odds without prematurely rounding the probability."""

    if american_odds is None:
        return None
    text = str(american_odds).strip().upper().replace("\u2212", "-")
    if text in {"EVEN", "EVENS", "EV", "EVS"}:
        return 0.5
    text = text.replace(",", "")
    try:
        odds = float(text)
    except (TypeError, ValueError):
        return None
    if odds == 0 or not math.isfinite(odds):
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def implied_probability(price: Any) -> Optional[float]:
    """Convert a price mapping, preferring the usually finer American quote."""

    if not isinstance(price, Mapping):
        return None
    probability = american_odds_to_probability(price.get("american"))
    if probability is not None:
        return clamp_probability(probability)
    decimal = price.get("decimal")
    if decimal is not None:
        try:
            decimal_value = float(str(decimal).replace(",", ""))
            if decimal_value > 1.0 and math.isfinite(decimal_value):
                return clamp_probability(1.0 / decimal_value)
        except (TypeError, ValueError):
            pass
    return None


def no_vig_two_way(over_probability: float, under_probability: float) -> float:
    denominator = over_probability + under_probability
    if denominator <= 0:
        raise ValueError("Two-way market has no valid probability mass")
    return clamp_probability(over_probability / denominator)


def fetch_json_payload(
    url: str,
    feed_name: str,
    validator: Callable[[Any], None],
    timeout: float = 30.0,
    retries: int = 3,
    extra_headers: Optional[Mapping[str, str]] = None,
) -> Any:
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if extra_headers:
        request_headers.update(extra_headers)
    request = Request(
        url,
        headers=request_headers,
    )

    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
            validator(payload)
            return payload
        except (
            HTTPError,
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
            # Retrying an authorization/WAF rejection against the identical
            # URL is not useful. Let the caller try its next endpoint version.
            if isinstance(exc, HTTPError) and exc.code in {401, 403, 404}:
                break
            if attempt < retries:
                time.sleep(0.75 * attempt)

    raise RuntimeError(f"Unable to load a valid {feed_name} payload: {last_error}")


def fetch_bovada_payload(timeout: float = 30.0, retries: int = 3) -> Any:
    params = urlencode(
        {"preMatchOnly": "true", "eventsLimit": "5000", "lang": "en"}
    )
    return fetch_json_payload(
        f"{BOVADA_ENDPOINT}?{params}",
        "Bovada NFL",
        validate_bovada_payload,
        timeout,
        retries,
    )


def fetch_underdog_via_colab_browser(endpoint: str, timeout: float) -> Any:
    """Fetch through the user's browser when Underdog blocks Colab's VM IP.

    Underdog explicitly permits cross-origin GETs. The payload is transferred
    to Python in bounded base64 chunks so Colab's message bridge never has to
    carry the full response in one reply.
    """

    try:
        from google.colab import output as colab_output
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Colab browser bridge is unavailable") from exc

    endpoint_literal = json.dumps(endpoint)
    fetch_script = f"""
        (async () => {{
          const response = await fetch({endpoint_literal}, {{
            method: 'GET',
            mode: 'cors',
            cache: 'no-store',
            headers: {{'Accept': 'application/json'}}
          }});
          if (!response.ok) {{
            throw new Error('Underdog HTTP ' + response.status);
          }}
          const buffer = await response.arrayBuffer();
          globalThis.__nflUnderdogPayloadBytes = new Uint8Array(buffer);
          return globalThis.__nflUnderdogPayloadBytes.length;
        }})()
    """

    try:
        byte_count = int(
            colab_output.eval_js(
                fetch_script,
                timeout_sec=max(60, int(math.ceil(timeout)) + 15),
            )
        )
        if byte_count <= 2:
            raise ValueError("browser returned an empty Underdog response")

        raw = bytearray()
        chunk_size = 512 * 1024
        for start in range(0, byte_count, chunk_size):
            end = min(start + chunk_size, byte_count)
            chunk_script = f"""
                (() => {{
                  const bytes = globalThis.__nflUnderdogPayloadBytes.subarray(
                    {start}, {end}
                  );
                  let binary = '';
                  const blockSize = 32768;
                  for (let i = 0; i < bytes.length; i += blockSize) {{
                    binary += String.fromCharCode(
                      ...bytes.subarray(i, Math.min(i + blockSize, bytes.length))
                    );
                  }}
                  return btoa(binary);
                }})()
            """
            encoded = colab_output.eval_js(chunk_script, timeout_sec=30)
            if not isinstance(encoded, str):
                raise ValueError("browser returned a non-text payload chunk")
            raw.extend(base64.b64decode(encoded, validate=True))

        if len(raw) != byte_count:
            raise ValueError(
                f"browser transfer was incomplete ({len(raw)} of {byte_count} bytes)"
            )
        payload = json.loads(bytes(raw).decode("utf-8"))
        validate_underdog_payload(payload)
        return payload
    except Exception as exc:
        raise RuntimeError(f"Colab browser fetch failed: {exc}") from exc
    finally:
        try:
            colab_output.eval_js(
                "delete globalThis.__nflUnderdogPayloadBytes",
                ignore_result=True,
                timeout_sec=5,
            )
        except Exception:
            pass


def fetch_underdog_payload(timeout: float = 45.0, retries: int = 3) -> Any:
    browser_headers = {
        "User-Agent": UNDERDOG_USER_AGENT,
        "Origin": "https://underdogfantasy.com",
        "Referer": "https://underdogfantasy.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }
    failures: List[str] = []
    for endpoint in UNDERDOG_ENDPOINTS:
        version_match = re.search(r"/beta/(v\d+)/", endpoint)
        version = version_match.group(1) if version_match else endpoint
        try:
            return fetch_json_payload(
                endpoint,
                f"Underdog {version}",
                validate_underdog_payload,
                timeout,
                retries,
                browser_headers,
            )
        except RuntimeError as exc:
            failures.append(str(exc))

    if running_in_google_colab():
        print(
            "Direct Underdog request failed; retrying through the Colab "
            "browser (this can take 20-60 seconds)...",
            file=sys.stderr,
            flush=True,
        )
        for endpoint in UNDERDOG_ENDPOINTS:
            try:
                return fetch_underdog_via_colab_browser(endpoint, timeout)
            except RuntimeError as exc:
                failures.append(str(exc))

    if failures and all("HTTP Error 403" in failure for failure in failures):
        raise RuntimeError(
            "all public endpoint versions returned HTTP 403; the provider's "
            "CDN may be rejecting this hosted runtime. Supply a browser-saved "
            "payload with --underdog-input, or run the script from a local IP."
        )
    raise RuntimeError("; ".join(failures))


# Backwards-compatible Bovada fetch helper.
def fetch_payload(timeout: float = 30.0, retries: int = 3) -> Any:
    return fetch_bovada_payload(timeout, retries)


def validate_bovada_payload(payload: Any) -> None:
    wrappers: List[Any]
    if isinstance(payload, list):
        wrappers = payload
    elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
        wrappers = [payload]
    else:
        raise ValueError(
            "Bovada returned an empty or unexpected response. The bare endpoint "
            "currently returns {}; keep the required query parameters enabled."
        )
    if not wrappers or not any(
        wrapper.get("events")
        for wrapper in wrappers
        if isinstance(wrapper, dict)
    ):
        raise ValueError("Bovada payload contains no NFL events")


def validate_underdog_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Underdog returned an unexpected non-object response")
    required_lists = ("players", "appearances", "games", "over_under_lines")
    missing = [key for key in required_lists if not isinstance(payload.get(key), list)]
    if missing:
        raise ValueError(
            "Underdog payload is missing expected lists: " + ", ".join(missing)
        )
    if not payload.get("over_under_lines"):
        raise ValueError("Underdog payload contains no over/under lines")


# Backwards-compatible Bovada validator.
def validate_payload(payload: Any) -> None:
    validate_bovada_payload(payload)


def iter_bovada_events(payload: Any) -> Iterable[Mapping[str, Any]]:
    validate_bovada_payload(payload)
    wrappers = payload if isinstance(payload, list) else [payload]
    for wrapper in wrappers:
        if not isinstance(wrapper, Mapping):
            continue
        for event in wrapper.get("events", []):
            if isinstance(event, Mapping):
                yield event


# Backwards-compatible Bovada iterator.
def iter_events(payload: Any) -> Iterable[Mapping[str, Any]]:
    yield from iter_bovada_events(payload)


def is_open_full_game_market(market: Mapping[str, Any]) -> bool:
    if str(market.get("status", "O")).upper() != "O":
        return False

    period = market.get("period")
    if isinstance(period, Mapping):
        abbreviation = str(period.get("abbreviation") or "").upper()
        description = str(period.get("description") or "").casefold()
        return bool(period.get("main")) or abbreviation == "G" or description == "game"

    # Fallback only for a future feed variant without structured period data.
    description = str(market.get("description") or "")
    partial_period = (
        r"(?:^|\W)(?:[1-4](?:ST|ND|RD|TH)?\s+QUARTER|Q[1-4]|[12]H|HALF)"
        r"(?:\W|$)"
    )
    return re.search(partial_period, description, re.I) is None


def is_open_outcome(outcome: Mapping[str, Any]) -> bool:
    return str(outcome.get("status", "O")).upper() == "O"


def split_player_team(label: Any) -> Tuple[str, str]:
    text = " ".join(str(label or "").split())
    match = re.match(r"^(.*?)\s*\(([A-Za-z]{2,4})\)\s*$", text)
    if match:
        return match.group(1).strip(), canonical_team(match.group(2))
    return text, "UNK"


def is_defense_name(name: str) -> bool:
    compact = " ".join(str(name or "").casefold().split())
    return bool(re.search(r"(?:\bd/st|\bdef/st|\bdefense)\s*$", compact))


def bovada_event_team_codes(event: Mapping[str, Any]) -> set[str]:
    """Resolve the two current teams from Bovada's event competitors."""

    teams: set[str] = set()
    competitors = event.get("competitors")
    if isinstance(competitors, list):
        for competitor in competitors:
            if not isinstance(competitor, Mapping):
                continue
            name = " ".join(str(competitor.get("name") or "").casefold().split())
            if name in NFL_TEAM_NAMES:
                teams.add(NFL_TEAM_NAMES[name])
    return teams


def normalized_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name).casefold()
    # Providers are inconsistent about generational suffixes (for example,
    # "Deebo Samuel Sr." versus "Deebo Samuel"). The game/team guard still
    # prevents this relaxed name key from merging unrelated players.
    decomposed = re.sub(r"\b(?:jr|sr|ii|iii|iv|v)\.?\s*$", "", decomposed)
    return re.sub(r"[^a-z0-9]+", "", decomposed)


def canonical_team(team: Any) -> str:
    code = re.sub(r"[^A-Za-z]", "", str(team or "")).upper() or "UNK"
    return TEAM_ALIASES.get(code, code)


def player_identity(event_id: str, name: str, team: str) -> Tuple[str, str, str]:
    return str(event_id), normalized_name(name), canonical_team(team)


def parse_iso_time_ms(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except (OverflowError, TypeError, ValueError):
        return None


def threshold_from_outcome(description: Any) -> Optional[float]:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*\+", str(description or ""))
    return float(match.group(1)) if match else None


def outcome_line(outcome: Mapping[str, Any]) -> Optional[float]:
    price = outcome.get("price")
    if isinstance(price, Mapping) and price.get("handicap") is not None:
        try:
            return float(str(price.get("handicap")).replace(",", ""))
        except (TypeError, ValueError):
            pass
    match = re.search(r"-?\d+(?:\.\d+)?", str(outcome.get("description") or ""))
    return float(match.group(0)) if match else None


def parse_two_way_market(
    outcomes: Any,
) -> Optional[Tuple[float, float]]:
    """Return (integer threshold, fair over probability)."""

    if not isinstance(outcomes, list):
        return None

    by_line: Dict[float, Dict[str, float]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping) or not is_open_outcome(outcome):
            continue
        description = str(outcome.get("description") or "").strip().casefold()
        if description.startswith("over"):
            side = "over"
        elif description.startswith("under"):
            side = "under"
        else:
            continue
        line = outcome_line(outcome)
        probability = implied_probability(outcome.get("price"))
        if line is not None and probability is not None:
            by_line.setdefault(line, {})[side] = probability

    paired = [
        (line, sides)
        for line, sides in by_line.items()
        if "over" in sides and "under" in sides
    ]
    if not paired:
        return None

    # Player props normally contain one line. If the feed supplies several,
    # choose the most balanced pair, which is usually the main line.
    line, sides = min(
        paired,
        key=lambda item: abs(
            no_vig_two_way(item[1]["over"], item[1]["under"]) - 0.5
        ),
    )
    threshold = float(math.floor(line) + 1)
    return threshold, no_vig_two_way(sides["over"], sides["under"])


def parse_bovada_payload(payload: Any) -> Dict[Tuple[str, str, str], PlayerMarkets]:
    players: Dict[Tuple[str, str, str], PlayerMarkets] = {}

    def get_player(event: Mapping[str, Any], label: Any) -> Optional[PlayerMarkets]:
        name, label_team = split_player_team(label)
        if not name or name.casefold() == "no touchdown scorer":
            return None
        # Defensive touchdown markets are not enough to project D/ST scoring.
        if is_defense_name(name):
            return None
        event_teams = bovada_event_team_codes(event)
        # Bovada sometimes leaves a former team or a college abbreviation in
        # parentheses on deep touchdown-scorer outcomes. Trust the suffix only
        # when it belongs to this event; Underdog appearance metadata can fill
        # an unknown team during the guarded same-player/same-game merge.
        team = label_team if not event_teams or label_team in event_teams else "UNK"
        event_id = str(event.get("id") or event.get("description") or "unknown-event")
        key = player_identity(event_id, name, team)
        if key not in players:
            start = event.get("startTime")
            try:
                start_ms = int(start) if start is not None else None
            except (TypeError, ValueError):
                start_ms = None
            players[key] = PlayerMarkets(
                event_id=event_id,
                name=name,
                team=team,
                matchup=str(event.get("description") or "Unknown matchup"),
                start_time_ms=start_ms,
            )
        return players[key]

    for event in iter_bovada_events(payload):
        if bool(event.get("live")):
            continue
        for display_group in event.get("displayGroups", []):
            if not isinstance(display_group, Mapping):
                continue
            for market in display_group.get("markets", []):
                if not isinstance(market, Mapping) or not is_open_full_game_market(market):
                    continue

                description = " ".join(str(market.get("description") or "").split())
                market_id = str(market.get("id") or "")

                # One-way non-passing touchdown ladders.
                td_threshold: Optional[int] = None
                if description.casefold() == "anytime touchdown scorer":
                    td_threshold = 1
                else:
                    td_match = re.fullmatch(
                        r"Player to Score\s+(\d+)\s+or More Touchdowns",
                        description,
                        flags=re.I,
                    )
                    if td_match:
                        td_threshold = int(td_match.group(1))

                if td_threshold is not None:
                    for outcome in market.get("outcomes", []):
                        if not isinstance(outcome, Mapping) or not is_open_outcome(outcome):
                            continue
                        probability = implied_probability(outcome.get("price"))
                        player = get_player(event, outcome.get("description"))
                        if probability is not None and player is not None:
                            player.market("any_touchdowns").add_alternate(
                                td_threshold, probability, market_id, "bovada"
                            )
                    continue

                match = re.fullmatch(
                    r"(Alternate|Total)\s+(.+?)\s+-\s+(.+)", description, flags=re.I
                )
                if not match:
                    continue

                family, raw_stat_name, player_label = match.groups()
                canonical_stat_name = next(
                    (
                        alias
                        for label, alias in STAT_ALIASES.items()
                        if raw_stat_name.casefold() == label.casefold()
                    ),
                    None,
                )
                if canonical_stat_name is None:
                    continue
                player = get_player(event, player_label)
                if player is None:
                    continue
                stat_market = player.market(canonical_stat_name)

                if family.casefold() == "alternate":
                    for outcome in market.get("outcomes", []):
                        if not isinstance(outcome, Mapping) or not is_open_outcome(outcome):
                            continue
                        threshold = threshold_from_outcome(outcome.get("description"))
                        probability = implied_probability(outcome.get("price"))
                        if threshold is not None and probability is not None:
                            stat_market.add_alternate(
                                threshold, probability, market_id, "bovada"
                            )
                else:
                    parsed = parse_two_way_market(market.get("outcomes"))
                    if parsed is not None:
                        threshold, fair_over_probability = parsed
                        stat_market.add_anchor(
                            threshold,
                            fair_over_probability,
                            market_id,
                            "bovada",
                        )

    return players


# Backwards-compatible name for callers that only use the Bovada parser.
def parse_payload(payload: Any) -> Dict[Tuple[str, str, str], PlayerMarkets]:
    return parse_bovada_payload(payload)


def mapping_by_id(items: Any) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(items, list):
        return {}
    return {
        str(item.get("id")): item
        for item in items
        if isinstance(item, Mapping) and item.get("id") is not None
    }


def underdog_option_probability(option: Mapping[str, Any]) -> Optional[float]:
    return implied_probability(
        {
            "american": option.get("american_price"),
            "decimal": option.get("decimal_price"),
        }
    )


def underdog_team_code(
    appearance: Mapping[str, Any], game: Mapping[str, Any], player: Mapping[str, Any]
) -> str:
    title = str(game.get("abbreviated_title") or game.get("title") or "")
    parts = re.split(r"\s+@\s+", title.strip(), maxsplit=1)
    if len(parts) != 2:
        return "UNK"
    away_code, home_code = (canonical_team(part) for part in parts)
    team_id = str(appearance.get("team_id") or player.get("team_id") or "")
    if team_id and team_id == str(game.get("away_team_id") or ""):
        return away_code
    if team_id and team_id == str(game.get("home_team_id") or ""):
        return home_code
    return "UNK"


def parse_underdog_payload(
    payload: Any,
) -> Dict[Tuple[str, str, str], PlayerMarkets]:
    """Parse active, pregame NFL player lines from Underdog's public feed."""

    validate_underdog_payload(payload)
    players_by_id = mapping_by_id(payload.get("players"))
    appearances_by_id = mapping_by_id(payload.get("appearances"))
    games_by_id = mapping_by_id(payload.get("games"))
    parsed_players: Dict[Tuple[str, str, str], PlayerMarkets] = {}

    # Keep roster metadata even when a player has only a Bovada touchdown
    # price. These empty records are used solely to repair stale Bovada team
    # suffixes and add positions during the cross-feed merge; make_projections
    # later skips any record that still has no supported market.
    for appearance in appearances_by_id.values():
        if str(appearance.get("type") or "").casefold() != "player":
            continue
        if str(appearance.get("match_type") or "").casefold() != "game":
            continue
        player = players_by_id.get(str(appearance.get("player_id")))
        game = games_by_id.get(str(appearance.get("match_id")))
        if not isinstance(player, Mapping) or not isinstance(game, Mapping):
            continue
        if str(player.get("sport_id") or "").upper() != "NFL":
            continue
        if str(game.get("sport_id") or "").upper() != "NFL":
            continue
        name = " ".join(
            part
            for part in (
                str(player.get("first_name") or "").strip(),
                str(player.get("last_name") or "").strip(),
            )
            if part
        )
        position = str(player.get("position_name") or "UNK").upper()
        if not name or is_defense_name(name) or position in {"DEF", "DST", "D/ST"}:
            continue
        team = underdog_team_code(appearance, game, player)
        game_id = str(game.get("id") or appearance.get("match_id") or "unknown")
        event_id = f"underdog:{game_id}"
        key = player_identity(event_id, name, team)
        parsed_players.setdefault(
            key,
            PlayerMarkets(
                event_id=event_id,
                name=name,
                team=team,
                matchup=str(
                    game.get("full_team_names_title")
                    or game.get("short_title")
                    or game.get("title")
                    or "Unknown matchup"
                ),
                start_time_ms=parse_iso_time_ms(game.get("scheduled_at")),
                position=position,
            ),
        )

    for line in payload.get("over_under_lines", []):
        if not isinstance(line, Mapping):
            continue
        if str(line.get("status") or "").casefold() != "active":
            continue
        if bool(line.get("live_event")):
            continue
        # Balanced lines are the ordinary, non-boosted market. Excluding
        # promotional line types avoids baking discounts into projections.
        if str(line.get("line_type") or "").casefold() != "balanced":
            continue

        over_under = line.get("over_under")
        if not isinstance(over_under, Mapping):
            continue
        if str(over_under.get("category") or "").casefold() != "player_prop":
            continue
        appearance_stat = over_under.get("appearance_stat")
        if not isinstance(appearance_stat, Mapping):
            continue
        stat = UNDERDOG_STAT_ALIASES.get(str(appearance_stat.get("stat") or ""))
        if stat is None:
            continue

        appearance = appearances_by_id.get(str(appearance_stat.get("appearance_id")))
        if not isinstance(appearance, Mapping):
            continue
        if str(appearance.get("type") or "").casefold() != "player":
            continue
        if str(appearance.get("match_type") or "").casefold() != "game":
            continue
        player = players_by_id.get(str(appearance.get("player_id")))
        game = games_by_id.get(str(appearance.get("match_id")))
        if not isinstance(player, Mapping) or not isinstance(game, Mapping):
            continue
        if str(player.get("sport_id") or "").upper() != "NFL":
            continue
        if str(game.get("sport_id") or "").upper() != "NFL":
            continue
        position = str(player.get("position_name") or "UNK").upper()

        try:
            line_value = float(str(line.get("stat_value")).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(line_value) or line_value < 0:
            continue
        # Higher than x.5 (or x with pushes possible) means an integer result
        # of floor(x) + 1 or greater, matching the survival-curve convention.
        threshold = float(math.floor(line_value) + 1)

        side_probabilities: Dict[str, List[float]] = {}
        for option in line.get("options", []):
            if not isinstance(option, Mapping):
                continue
            if str(option.get("status") or "").casefold() != "active":
                continue
            side = str(option.get("choice") or "").casefold()
            if side not in {"higher", "lower"}:
                continue
            probability = underdog_option_probability(option)
            if probability is not None:
                side_probabilities.setdefault(side, []).append(probability)

        higher_values = side_probabilities.get("higher", [])
        lower_values = side_probabilities.get("lower", [])
        if not higher_values:
            # A lower-only selection cannot be converted with the same
            # one-way vig adjustment, so omit it rather than invert it badly.
            continue
        higher_probability = statistics.median(higher_values)

        name = " ".join(
            part
            for part in (
                str(player.get("first_name") or "").strip(),
                str(player.get("last_name") or "").strip(),
            )
            if part
        )
        if not name or is_defense_name(name) or position in {"DEF", "DST", "D/ST"}:
            continue
        team = underdog_team_code(appearance, game, player)
        game_id = str(game.get("id") or appearance.get("match_id") or "unknown")
        event_id = f"underdog:{game_id}"
        key = player_identity(event_id, name, team)
        if key not in parsed_players:
            parsed_players[key] = PlayerMarkets(
                event_id=event_id,
                name=name,
                team=team,
                matchup=str(
                    game.get("full_team_names_title")
                    or game.get("short_title")
                    or game.get("title")
                    or "Unknown matchup"
                ),
                start_time_ms=parse_iso_time_ms(game.get("scheduled_at")),
                position=position,
            )

        market = parsed_players[key].market(stat)
        market_id = str(line.get("id") or line.get("stable_id") or "")
        if lower_values:
            lower_probability = statistics.median(lower_values)
            market.add_anchor(
                threshold,
                no_vig_two_way(higher_probability, lower_probability),
                market_id,
                "underdog",
            )
        else:
            market.add_alternate(
                threshold, higher_probability, market_id, "underdog"
            )

    return parsed_players


def merge_game_token(player: PlayerMarkets) -> str:
    if player.start_time_ms is not None:
        # Feeds occasionally differ by a few seconds. Minute precision is
        # strict enough to separate NFL games while tolerating that drift.
        return f"time:{int(round(player.start_time_ms / 60000.0))}"
    return "matchup:" + normalized_name(player.matchup)


def merge_player_data(target: PlayerMarkets, incoming: PlayerMarkets) -> None:
    if target.team == "UNK" and incoming.team != "UNK":
        target.team = incoming.team
    if target.position == "UNK" and incoming.position != "UNK":
        target.position = incoming.position
    if target.start_time_ms is None:
        target.start_time_ms = incoming.start_time_ms
    if len(incoming.matchup) > len(target.matchup):
        target.matchup = incoming.matchup

    for stat, incoming_market in incoming.stats.items():
        target_market = target.market(stat)
        for threshold, probabilities in incoming_market.alternate.items():
            target_market.alternate.setdefault(threshold, []).extend(probabilities)
        target_market.anchors.extend(incoming_market.anchors)
        target_market.market_ids.update(incoming_market.market_ids)
        target_market.providers.update(incoming_market.providers)


def merge_player_collections(
    *collections: Mapping[Tuple[str, str, str], PlayerMarkets],
) -> Dict[Tuple[str, str, str], PlayerMarkets]:
    """Merge provider records for the same player in the same scheduled game."""

    merged_list: List[PlayerMarkets] = []
    loose_index: Dict[Tuple[str, str], List[PlayerMarkets]] = {}

    for collection in collections:
        for incoming in collection.values():
            loose_key = (merge_game_token(incoming), normalized_name(incoming.name))
            candidates = loose_index.get(loose_key, [])
            incoming_team = canonical_team(incoming.team)
            compatible = [
                candidate
                for candidate in candidates
                if canonical_team(candidate.team) == incoming_team
                or "UNK" in {canonical_team(candidate.team), incoming_team}
            ]
            target = compatible[0] if len(compatible) == 1 else None
            if target is None:
                incoming.team = incoming_team
                merged_list.append(incoming)
                loose_index.setdefault(loose_key, []).append(incoming)
            else:
                merge_player_data(target, incoming)

    result: Dict[Tuple[str, str, str], PlayerMarkets] = {}
    for index, player in enumerate(merged_list):
        key = player_identity(player.event_id, player.name, player.team)
        if key in result:
            key = (f"{player.event_id}:{index}", key[1], key[2])
        result[key] = player
    return result


def isotonic_nonincreasing(
    points: Sequence[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """Weighted pool-adjacent-violators fit for a survival curve."""

    if not points:
        return []
    ordered = sorted(points, key=lambda point: point[0])
    blocks: List[List[float]] = []
    for index, (_, probability, weight) in enumerate(ordered):
        blocks.append([float(index), float(index), weight, probability * weight])
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = previous[3] / previous[2]
            current_mean = current[3] / current[2]
            if previous_mean >= current_mean:
                break
            merged = [
                previous[0],
                current[1],
                previous[2] + current[2],
                previous[3] + current[3],
            ]
            blocks[-2:] = [merged]

    fitted = [0.0] * len(ordered)
    for start, end, weight, weighted_sum in blocks:
        mean = clamp_probability(weighted_sum / weight)
        for index in range(int(start), int(end) + 1):
            fitted[index] = mean
    return [
        (ordered[index][0], fitted[index], ordered[index][2])
        for index in range(len(ordered))
    ]


def collapsed_alternate(market: StatMarket) -> List[Tuple[float, float, float]]:
    points = [
        (threshold, statistics.median(probabilities), float(len(probabilities)))
        for threshold, probabilities in market.alternate.items()
        if probabilities
    ]
    return isotonic_nonincreasing(points)


def interpolate_logit(
    points: Sequence[Tuple[float, float, float]], threshold: float
) -> Optional[float]:
    ordered = sorted(points, key=lambda point: point[0])
    for point_threshold, probability, _ in ordered:
        if math.isclose(point_threshold, threshold, abs_tol=1e-9):
            return probability
    for left, right in zip(ordered, ordered[1:]):
        x0, p0, _ = left
        x1, p1, _ = right
        if x0 <= threshold <= x1 and x1 > x0:
            weight = (threshold - x0) / (x1 - x0)
            return logistic(logit(p0) + weight * (logit(p1) - logit(p0)))
    return None


def local_logit_vig_shift(market: StatMarket) -> Optional[float]:
    alternate = collapsed_alternate(market)
    shifts = []
    for anchor in market.anchors:
        raw_probability = interpolate_logit(alternate, anchor.threshold)
        if raw_probability is not None:
            shifts.append(logit(raw_probability) - logit(anchor.probability))
    if not shifts:
        return None
    return min(0.75, max(-0.35, statistics.median(shifts)))


def estimate_global_logit_vig(
    players: Mapping[Tuple[str, str, str], PlayerMarkets],
    fallback: float = DEFAULT_FALLBACK_LOGIT_VIG,
) -> Tuple[float, int]:
    shifts = [
        shift
        for player in players.values()
        for market in player.stats.values()
        if (shift := local_logit_vig_shift(market)) is not None
    ]
    if len(shifts) >= 3:
        return statistics.median(shifts), len(shifts)
    return fallback, len(shifts)


def fair_observations(
    market: StatMarket, global_logit_vig: float
) -> Tuple[List[Observation], str]:
    alternate = collapsed_alternate(market)
    local_shift = local_logit_vig_shift(market)
    shift = local_shift if local_shift is not None else global_logit_vig

    observations = [
        Observation(
            threshold=threshold,
            probability=logistic(logit(probability) - shift),
            weight=weight,
            source="alternate",
        )
        for threshold, probability, weight in alternate
    ]
    observations.extend(market.anchors)

    # Aggregate coincident total and alternate thresholds, then impose the
    # required non-increasing shape one final time.
    by_threshold: Dict[float, List[Observation]] = {}
    for observation in observations:
        by_threshold.setdefault(observation.threshold, []).append(observation)
    combined = []
    for threshold, entries in by_threshold.items():
        total_weight = sum(entry.weight for entry in entries)
        probability = sum(
            entry.probability * entry.weight for entry in entries
        ) / total_weight
        combined.append((threshold, probability, total_weight))
    monotone = isotonic_nonincreasing(combined)

    provider_label = "+".join(sorted(market.providers)) or "unknown"
    if alternate and market.anchors:
        source = f"{provider_label}:total+alternate"
    elif market.anchors:
        source = f"{provider_label}:total-only"
    else:
        source = f"{provider_label}:alternate-only"
    return [Observation(x, p, w, source) for x, p, w in monotone], source


def fit_weibull(
    observations: Sequence[Observation], default_shape: float
) -> Distribution:
    valid = [
        observation
        for observation in observations
        if observation.threshold > 0
        and EPSILON < observation.probability < 1.0 - EPSILON
    ]
    if not valid:
        return Distribution("weibull", 0.0, default_shape, EPSILON)

    shape = default_shape
    if len(valid) >= 2 and len({item.threshold for item in valid}) >= 2:
        x_values = [math.log(item.threshold) for item in valid]
        y_values = [math.log(-math.log(item.probability)) for item in valid]
        weights = [item.weight for item in valid]
        weight_sum = sum(weights)
        x_mean = sum(x * w for x, w in zip(x_values, weights)) / weight_sum
        y_mean = sum(y * w for y, w in zip(y_values, weights)) / weight_sum
        denominator = sum(
            w * (x - x_mean) ** 2 for x, w in zip(x_values, weights)
        )
        if denominator > 0:
            fitted_shape = sum(
                w * (x - x_mean) * (y - y_mean)
                for x, y, w in zip(x_values, y_values, weights)
            ) / denominator
            if math.isfinite(fitted_shape) and fitted_shape > 0:
                shape = min(8.0, max(0.70, fitted_shape))

    # Refit the intercept after shape clamping.
    weights = [item.weight for item in valid]
    weight_sum = sum(weights)
    intercept = sum(
        item.weight
        * (math.log(-math.log(item.probability)) - shape * math.log(item.threshold))
        for item in valid
    ) / weight_sum
    scale = math.exp(-intercept / shape)
    mean = scale * math.gamma(1.0 + 1.0 / shape)
    if not math.isfinite(mean) or mean < 0:
        mean = 0.0
    return Distribution("weibull", mean, shape, scale)


def poisson_survival(threshold: int, rate: float) -> float:
    """P[X >= threshold] for X ~ Poisson(rate)."""

    if threshold <= 0:
        return 1.0
    if rate <= 0:
        return 0.0
    term = math.exp(-rate)
    cumulative = term
    for value in range(1, threshold):
        term *= rate / value
        cumulative += term
    return clamp_probability(1.0 - cumulative)


def fit_poisson(observations: Sequence[Observation]) -> Distribution:
    valid = [
        observation
        for observation in observations
        if observation.threshold >= 1
        and EPSILON < observation.probability < 1.0 - EPSILON
    ]
    if not valid:
        return Distribution("poisson", 0.0, 0.0)

    max_threshold = max(int(round(item.threshold)) for item in valid)
    lower_log = math.log(1e-4)
    upper_log = math.log(max(12.0, max_threshold * 5.0))

    def objective(log_rate: float) -> float:
        rate = math.exp(log_rate)
        total = 0.0
        for item in valid:
            fitted = poisson_survival(int(round(item.threshold)), rate)
            residual = logit(fitted) - logit(item.probability)
            total += item.weight * residual * residual
        return total

    # Golden-section minimization in log(rate), avoiding a SciPy dependency.
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = lower_log, upper_log
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(100):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    rate = math.exp((left + right) / 2.0)
    return Distribution("poisson", rate, rate)


def fit_stat_distribution(
    stat: str, market: StatMarket, global_logit_vig: float
) -> Tuple[Distribution, str]:
    observations, source = fair_observations(market, global_logit_vig)
    if stat in YARD_STATS:
        return fit_weibull(observations, DEFAULT_WEIBULL_SHAPES[stat]), source
    if stat in COUNT_STATS:
        return fit_poisson(observations), source
    raise ValueError(f"Unsupported stat: {stat}")


REQUIRED_PROJECTION_COMPONENTS = {
    "QB": {"passing_yards", "passing_touchdowns", "interceptions",
           "rushing_yards", "any_touchdowns"},
    "RB": {"rushing_yards", "receiving_yards", "receptions", "any_touchdowns"},
    "WR": {"receiving_yards", "receptions", "any_touchdowns"},
    "TE": {"receiving_yards", "receptions", "any_touchdowns"},
}


def missing_projection_components(means, position):
    """Require position-specific core stats; absence is not an observed zero.

    Rare ancillary stats (e.g. WR rushing and lost fumbles) still contribute
    when available, but are not required for the core coverage label.
    """
    required = REQUIRED_PROJECTION_COMPONENTS.get(str(position).upper())
    if required is None:
        return ["known position"]
    return sorted(stat for stat in required
                  if stat not in means or not math.isfinite(means[stat]))


def projection_quality(
    distributions: Mapping[str, Distribution],
    sources: Mapping[str, str],
    position: str = "UNK",
) -> str:
    means = {stat: dist.mean for stat, dist in distributions.items()}
    # TD-only regression is a separately labelled estimate, never a complete
    # direct component sum. Preserve that route for non-quarterbacks.
    if set(means) == {"any_touchdowns"} and str(position).upper() != "QB":
        return "td-only" if math.isfinite(means["any_touchdowns"]) else "partial"
    if missing_projection_components(means, position):
        return "partial"
    required = REQUIRED_PROJECTION_COMPONENTS[str(position).upper()]
    has_total_anchor = any("total" in sources.get(stat, "") for stat in required)
    return "good" if has_total_anchor else "fair"


def iso_start_time(start_time_ms: Optional[int]) -> Optional[str]:
    if start_time_ms is None:
        return None
    try:
        return datetime.fromtimestamp(
            start_time_ms / 1000.0, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _ordinary_least_squares(
    samples: Sequence[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Return intercept and slope for y = intercept + slope*x."""

    if len(samples) < 2:
        return None
    x_mean = sum(x for x, _ in samples) / len(samples)
    y_mean = sum(y for _, y in samples) / len(samples)
    denominator = sum((x - x_mean) ** 2 for x, _ in samples)
    if denominator <= EPSILON:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in samples) / denominator
    intercept = y_mean - slope * x_mean
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        return None
    return intercept, slope


def fit_td_regression(
    samples: Sequence[Tuple[float, float]],
    label: str,
    minimum_samples: int,
) -> Optional[TdRegression]:
    """Fit a robust linear regression after one MAD residual trim."""

    clean = [
        (float(expected_td), float(fantasy_points))
        for expected_td, fantasy_points in samples
        if math.isfinite(expected_td)
        and math.isfinite(fantasy_points)
        and expected_td > 0
        and fantasy_points > 0
    ]
    if len(clean) < minimum_samples:
        return None

    coefficients = _ordinary_least_squares(clean)
    if coefficients is None:
        return None
    intercept, slope = coefficients
    residuals = [y - (intercept + slope * x) for x, y in clean]
    residual_median = statistics.median(residuals)
    residual_mad = statistics.median(
        abs(residual - residual_median) for residual in residuals
    )
    if residual_mad > EPSILON:
        robust_sigma = 1.4826 * residual_mad
        trimmed = [
            sample
            for sample, residual in zip(clean, residuals)
            if abs(residual - residual_median)
            <= TD_REGRESSION_RESIDUAL_MAD_LIMIT * robust_sigma
        ]
        if minimum_samples <= len(trimmed) < len(clean):
            trimmed_coefficients = _ordinary_least_squares(trimmed)
            if trimmed_coefficients is not None:
                clean = trimmed
                intercept, slope = trimmed_coefficients

    predictions = [intercept + slope * x for x, _ in clean]
    y_mean = sum(y for _, y in clean) / len(clean)
    residual_sum_squares = sum(
        (y - fitted) ** 2 for (_, y), fitted in zip(clean, predictions)
    )
    total_sum_squares = sum((y - y_mean) ** 2 for _, y in clean)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > EPSILON
        else 0.0
    )
    return TdRegression(
        label=label,
        intercept=intercept,
        slope=slope,
        sample_count=len(clean),
        r_squared=max(0.0, min(1.0, r_squared)),
        maximum_target=max(y for _, y in clean),
    )


def fit_td_regression_models(
    projections: Sequence[Projection],
) -> Dict[str, TdRegression]:
    """Train global and position fits from good non-QB projections only."""

    global_samples: List[Tuple[float, float]] = []
    position_samples: Dict[str, List[Tuple[float, float]]] = {}
    for projection in projections:
        stats = set(projection.stat_means)
        if (
            projection.quality != "good"
            or "any_touchdowns" not in stats
            or projection.position.upper() == "QB"
            or bool(stats & {"passing_yards", "passing_touchdowns"})
        ):
            continue
        sample = (
            projection.stat_means["any_touchdowns"],
            projection.fantasy_points,
        )
        global_samples.append(sample)
        position = projection.position.upper()
        if position in {"RB", "WR", "TE"}:
            position_samples.setdefault(position, []).append(sample)

    models: Dict[str, TdRegression] = {}
    global_model = fit_td_regression(
        global_samples,
        "skill",
        MIN_GLOBAL_TD_REGRESSION_SAMPLES,
    )
    if global_model is not None:
        models["*"] = global_model
    for position, samples in position_samples.items():
        model = fit_td_regression(
            samples,
            position,
            MIN_POSITION_TD_REGRESSION_SAMPLES,
        )
        if model is not None:
            models[position] = model
    return models


def estimate_td_only_fantasy_points(
    projections: Sequence[Projection], scoring: Scoring
) -> None:
    """Replace bare TD scoring with slate-regressed full-FP estimates in place."""

    models = fit_td_regression_models(projections)
    global_model = models.get("*")
    for projection in projections:
        if (
            projection.quality != "td-only"
            or "any_touchdowns" not in projection.stat_means
        ):
            continue
        position = projection.position.upper()
        model = None if position == "QB" else models.get(position, global_model)
        expected_touchdowns = projection.stat_means.get("any_touchdowns", 0.0)
        if model is None:
            projection.fantasy_points_method = "touchdown-component-only"
            continue
        projection.fantasy_points = round(
            model.predict(
                expected_touchdowns,
                scoring.rushing_receiving_touchdown,
            ),
            2,
        )
        projection.quality = "td-estimate"
        projection.fantasy_points_method = (
            f"td-regression:{model.label};n={model.sample_count};"
            f"r2={model.r_squared:.3f};intercept={model.intercept:.3f};"
            f"slope={model.slope:.3f}"
        )


def make_projections(
    players: Mapping[Tuple[str, str, str], PlayerMarkets],
    scoring: Scoring,
    fallback_logit_vig: float = DEFAULT_FALLBACK_LOGIT_VIG,
) -> Tuple[List[Projection], float, int]:
    global_logit_vig, calibration_pairs = estimate_global_logit_vig(
        players, fallback_logit_vig
    )
    projections: List[Projection] = []

    for player in players.values():
        distributions: Dict[str, Distribution] = {}
        sources: Dict[str, str] = {}
        for stat, market in player.stats.items():
            if not market.alternate and not market.anchors:
                continue
            distribution, source = fit_stat_distribution(
                stat, market, global_logit_vig
            )
            distributions[stat] = distribution
            sources[stat] = source

        if not distributions:
            continue
        means = {stat: distribution.mean for stat, distribution in distributions.items()}
        fantasy_points = (
            means.get("passing_yards", 0.0) * scoring.passing_yard
            + means.get("rushing_yards", 0.0) * scoring.rushing_yard
            + means.get("receiving_yards", 0.0) * scoring.receiving_yard
            + means.get("receptions", 0.0) * scoring.reception
            + means.get("passing_touchdowns", 0.0) * scoring.passing_touchdown
            + means.get("any_touchdowns", 0.0)
            * scoring.rushing_receiving_touchdown
            + means.get("interceptions", 0.0) * scoring.interception
            + means.get("fumbles_lost", 0.0) * scoring.fumble_lost
        )
        projections.append(
            Projection(
                event_id=player.event_id,
                matchup=player.matchup,
                start_time_utc=iso_start_time(player.start_time_ms),
                team=player.team,
                player=player.name,
                position=player.position,
                fantasy_points=round(fantasy_points, 2),
                quality=projection_quality(distributions, sources, player.position),
                stat_means={key: round(value, 3) for key, value in means.items()},
                sources=sources,
            )
        )

    estimate_td_only_fantasy_points(projections, scoring)
    projections.sort(
        key=lambda item: (
            item.start_time_utc or "9999",
            item.matchup,
            item.team,
            -item.fantasy_points,
            item.player,
        )
    )
    return projections, global_logit_vig, calibration_pairs


def print_projections(
    projections: Sequence[Projection],
    scoring_name: str,
    logit_vig: float,
    calibration_pairs: int,
    include_td_only: bool,
    loaded_feeds: Sequence[str],
) -> None:
    print(
        f"Feeds: {' + '.join(loaded_feeds)} | Scoring: {scoring_name} | "
        f"one-way logit adjustment: {logit_vig:.3f} "
        f"({calibration_pairs} local calibration pairs)"
    )
    print("Coverage labels describe available prop components, not certainty.")
    estimated_count = sum(
        projection.quality == "td-estimate" for projection in projections
    )
    if estimated_count:
        print(
            f"TD-only regression estimates: {estimated_count}; targets are good "
            "non-QB projections from this slate."
        )

    visible = [
        projection
        for projection in projections
        if include_td_only
        or projection.quality not in {"td-only", "td-estimate"}
    ]
    if not visible:
        print("No sufficiently covered player prop projections were found.")
        return

    current_group: Optional[Tuple[str, str, str]] = None
    for projection in visible:
        group = (
            projection.start_time_utc or "time unavailable",
            projection.matchup,
            projection.team,
        )
        if group != current_group:
            print(
                f"\n{projection.matchup} | {projection.start_time_utc or 'time unavailable'}"
            )
            print(f"Team: {projection.team}")
            current_group = group

        component_order = [
            "passing_yards",
            "rushing_yards",
            "receiving_yards",
            "receptions",
            "passing_touchdowns",
            "any_touchdowns",
            "interceptions",
            "fumbles_lost",
        ]
        components = " ".join(
            f"{STAT_LABELS[stat]}={projection.stat_means[stat]:.2f}"
            for stat in component_order
            if stat in projection.stat_means
        )
        regression_note = ""
        if projection.quality == "td-estimate":
            match = re.search(
                r"^td-regression:([^;]+);n=(\d+);r2=([\d.]+)",
                projection.fantasy_points_method,
            )
            if match:
                label, sample_count, r_squared = match.groups()
                regression_note = (
                    f" | model={label} n={sample_count} R2={r_squared}"
                )
        print(
            f"  {projection.player}: {projection.fantasy_points:.2f} FP "
            f"[{projection.quality}] | {components}{regression_note}"
        )

    hidden_count = len(projections) - len(visible)
    if hidden_count:
        print(
            f"\nHidden: {hidden_count} touchdown-only estimates. "
            "Remove --exclude-td-only to display them."
        )


def run_self_test() -> None:
    assert math.isclose(american_odds_to_probability("EVEN") or 0, 0.5)
    assert math.isclose(american_odds_to_probability("+100") or 0, 0.5)
    assert math.isclose(
        american_odds_to_probability("-110") or 0, 110 / 210, rel_tol=1e-12
    )
    assert math.isclose(no_vig_two_way(110 / 210, 110 / 210), 0.5)
    assert canonical_team("NOR") == "NO"
    assert is_defense_name("Seattle Seahawks Def/ST")
    assert is_defense_name("CHI Bears D/ST")
    assert bovada_event_team_codes(
        {
            "competitors": [
                {"name": "New England Patriots"},
                {"name": "Seattle Seahawks"},
            ]
        }
    ) == {"NE", "SEA"}

    monotone = isotonic_nonincreasing(
        [(10, 0.80, 1), (20, 0.60, 1), (30, 0.62, 1), (40, 0.20, 1)]
    )
    assert all(
        left[1] >= right[1] for left, right in zip(monotone, monotone[1:])
    )

    rate = 3.25
    synthetic = [
        Observation(k, poisson_survival(k, rate), 1.0, "test")
        for k in range(1, 6)
    ]
    fitted = fit_poisson(synthetic)
    assert math.isclose(fitted.mean, rate, rel_tol=1e-5)

    weibull = fit_weibull(
        [
            Observation(40, math.exp(-((40 / 80) ** 2)), 1.0, "test"),
            Observation(80, math.exp(-1), 1.0, "test"),
            Observation(120, math.exp(-((120 / 80) ** 2)), 1.0, "test"),
        ],
        default_shape=2.0,
    )
    assert math.isclose(weibull.parameter_1, 2.0, rel_tol=1e-5)
    assert math.isclose(weibull.parameter_2 or 0, 80.0, rel_tol=1e-5)

    def synthetic_underdog_line(
        line_id: str,
        stat: str,
        value: str,
        options: Sequence[Tuple[str, str]],
    ) -> Dict[str, Any]:
        return {
            "id": line_id,
            "line_type": "balanced",
            "live_event": False,
            "status": "active",
            "stat_value": value,
            "options": [
                {
                    "choice": choice,
                    "american_price": price,
                    "status": "active",
                }
                for choice, price in options
            ],
            "over_under": {
                "category": "player_prop",
                "appearance_stat": {"appearance_id": "a1", "stat": stat},
            },
        }

    underdog_fixture = {
        "players": [
            {
                "id": "p1",
                "first_name": "Test",
                "last_name": "Quarterback",
                "position_name": "QB",
                "sport_id": "NFL",
                "team_id": "away-id",
            }
        ],
        "appearances": [
            {
                "id": "a1",
                "match_id": 7,
                "match_type": "Game",
                "player_id": "p1",
                "team_id": "away-id",
                "type": "Player",
            }
        ],
        "games": [
            {
                "id": 7,
                "abbreviated_title": "NE @ SEA",
                "away_team_id": "away-id",
                "home_team_id": "home-id",
                "full_team_names_title": "New England Patriots @ Seattle Seahawks",
                "scheduled_at": "2026-09-10T00:20:00Z",
                "sport_id": "NFL",
            }
        ],
        "over_under_lines": [
            synthetic_underdog_line(
                "line-yards",
                "passing_yds",
                "249.5",
                (("higher", "-110"), ("lower", "-110")),
            ),
            synthetic_underdog_line(
                "line-fumble", "fumbles_lost", "0.5", (("higher", "+350"),)
            ),
        ],
    }
    parsed_underdog = parse_underdog_payload(underdog_fixture)
    assert len(parsed_underdog) == 1
    test_player = next(iter(parsed_underdog.values()))
    assert test_player.team == "NE" and test_player.position == "QB"
    assert math.isclose(
        test_player.stats["passing_yards"].anchors[0].probability, 0.5
    )
    assert test_player.stats["passing_yards"].anchors[0].threshold == 250.0
    assert test_player.stats["fumbles_lost"].alternate[1.0]

    bovada_record = PlayerMarkets(
        event_id="bovada:7",
        name="A.J. Brown",
        team="UNK",
        matchup="New England Patriots @ Seattle Seahawks",
        start_time_ms=parse_iso_time_ms("2026-09-10T00:20:00Z"),
    )
    bovada_record.market("receiving_yards").add_alternate(
        50, 0.65, "b1", "bovada"
    )
    underdog_record = PlayerMarkets(
        event_id="underdog:7",
        name="AJ Brown",
        team="NE",
        matchup="NE @ SEA",
        start_time_ms=parse_iso_time_ms("2026-09-10T00:20:00Z"),
        position="WR",
    )
    underdog_record.market("receptions").add_anchor(
        5, 0.5, "u1", "underdog"
    )
    merged = merge_player_collections(
        {player_identity("bovada:7", "A.J. Brown", "NE"): bovada_record},
        {player_identity("underdog:7", "AJ Brown", "NE"): underdog_record},
    )
    assert len(merged) == 1
    merged_player = next(iter(merged.values()))
    assert set(merged_player.stats) == {"receiving_yards", "receptions"}
    assert merged_player.team == "NE"
    assert merged_player.position == "WR"

    regression_training = [
        Projection(
            event_id="training",
            matchup="Training @ Sample",
            start_time_utc=None,
            team="TST",
            player=f"Training Receiver {index}",
            position="WR",
            fantasy_points=4.0 + 15.0 * expected_td,
            quality="good",
            stat_means={
                "receiving_yards": 40.0,
                "receptions": 3.0,
                "any_touchdowns": expected_td,
            },
            sources={},
        )
        for index, expected_td in enumerate(
            (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
            1,
        )
    ]
    td_only_projection = Projection(
        event_id="test",
        matchup="Test @ Sample",
        start_time_utc=None,
        team="TST",
        player="Touchdown Only",
        position="UNK",
        fantasy_points=3.0,
        quality="td-only",
        stat_means={"any_touchdowns": 0.50},
        sources={"any_touchdowns": "bovada:alternate-only"},
    )
    regression_sample = regression_training + [td_only_projection]
    models = fit_td_regression_models(regression_sample)
    assert "*" in models and "WR" in models
    assert math.isclose(models["*"].intercept, 4.0, rel_tol=1e-10)
    assert math.isclose(models["*"].slope, 15.0, rel_tol=1e-10)
    estimate_td_only_fantasy_points(regression_sample, SCORING_PRESETS["yahoo"])
    assert td_only_projection.quality == "td-estimate"
    assert math.isclose(td_only_projection.fantasy_points, 11.5, rel_tol=1e-10)
    assert td_only_projection.fantasy_points_method.startswith(
        "td-regression:skill;n=12;"
    )
    assert parse_cli_args([]).include_td_only is True
    assert parse_cli_args(["--exclude-td-only"]).include_td_only is False
    print("Self-test passed")


def load_json_file(
    path: Path, validator: Callable[[Any], None], feed_name: str
) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        validator(payload)
    except ValueError as exc:
        raise ValueError(f"{feed_name} input {path}: {exc}") from exc
    return payload


def load_bovada_payload(path: Optional[Path]) -> Any:
    if path is None:
        return fetch_bovada_payload()
    return load_json_file(path, validate_bovada_payload, "Bovada")


def load_underdog_payload(path: Optional[Path]) -> Any:
    if path is None:
        return fetch_underdog_payload()
    return load_json_file(path, validate_underdog_payload, "Underdog")


# Backwards-compatible saved/live Bovada loader.
def load_payload(path: Optional[Path]) -> Any:
    return load_bovada_payload(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build market-implied NFL fantasy projections from paired "
            "Underdog props and Bovada alternate lines."
        )
    )
    parser.add_argument(
        "--source",
        choices=("hybrid", "bovada", "underdog"),
        default="hybrid",
        help="Data feed selection (default: hybrid).",
    )
    parser.add_argument(
        "--input",
        "--bovada-input",
        dest="bovada_input",
        type=Path,
        help=(
            "Read a saved Bovada JSON payload instead of requesting it live. "
            "--input remains as a backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--underdog-input",
        type=Path,
        help="Read a saved Underdog JSON payload instead of requesting it live.",
    )
    parser.add_argument(
        "--scoring",
        choices=sorted(SCORING_PRESETS),
        default="yahoo",
        help="Fantasy scoring preset (default: yahoo).",
    )
    parser.add_argument(
        "--fallback-logit-vig",
        type=float,
        default=DEFAULT_FALLBACK_LOGIT_VIG,
        help=(
            "Fallback log-odds adjustment for one-way prices when fewer than "
            "three local total/alternate pairs exist (default: 0.17)."
        ),
    )
    td_display = parser.add_mutually_exclusive_group()
    td_display.add_argument(
        "--include-td-only",
        dest="include_td_only",
        action="store_true",
        help=(
            "Show touchdown-only regression estimates (enabled by default; "
            "retained for backwards compatibility)."
        ),
    )
    td_display.add_argument(
        "--exclude-td-only",
        dest="include_td_only",
        action="store_false",
        help="Hide touchdown-only regression estimates.",
    )
    parser.set_defaults(include_td_only=True)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON rather than the formatted report.",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run deterministic math checks and exit."
    )
    return parser


def running_in_notebook_kernel() -> bool:
    return (
        "ipykernel" in sys.modules
        or running_in_google_colab()
        or Path(sys.argv[0]).name == "ipykernel_launcher.py"
    )


def running_in_google_colab() -> bool:
    return (
        "google.colab" in sys.modules
        or Path(sys.argv[0]).name == "colab_kernel_launcher.py"
    )


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse user options without treating Jupyter's kernel file as an option.

    Colab/IPython launches the notebook process with ``-f kernel-....json``.
    When a complete script is pasted into a cell, that process-level argument
    is still present in ``sys.argv``.  Remove only that exact connection-file
    pair; all other unknown arguments continue to raise an argparse error.
    """

    parser = build_parser()
    if argv is not None:
        return parser.parse_args(list(argv))

    raw_arguments = list(sys.argv[1:])
    if not running_in_notebook_kernel():
        return parser.parse_args(raw_arguments)

    cleaned_arguments: List[str] = []
    index = 0
    while index < len(raw_arguments):
        argument = raw_arguments[index]
        if argument == "-f" and index + 1 < len(raw_arguments):
            connection_file = raw_arguments[index + 1]
            if re.search(r"(?:^|[/\\])kernel-[^/\\]+\.json$", connection_file):
                index += 2
                continue
        if argument.startswith("-f="):
            connection_file = argument[3:]
            if re.search(r"(?:^|[/\\])kernel-[^/\\]+\.json$", connection_file):
                index += 1
                continue
        cleaned_arguments.append(argument)
        index += 1

    return parser.parse_args(cleaned_arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_cli_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    if not math.isfinite(args.fallback_logit_vig):
        print("Error: --fallback-logit-vig must be finite", file=sys.stderr)
        return 1

    collections: List[Mapping[Tuple[str, str, str], PlayerMarkets]] = []
    loaded_feeds: List[str] = []
    warnings: List[str] = []

    requested_feeds = (
        ("bovada", "underdog") if args.source == "hybrid" else (args.source,)
    )
    for feed in requested_feeds:
        try:
            if feed == "bovada":
                payload = load_bovada_payload(args.bovada_input)
                parsed = parse_bovada_payload(payload)
                display_name = "Bovada"
            else:
                payload = load_underdog_payload(args.underdog_input)
                parsed = parse_underdog_payload(payload)
                display_name = "Underdog"
            if not parsed:
                raise ValueError("no supported pregame NFL player props were found")
            collections.append(parsed)
            loaded_feeds.append(display_name)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"{feed.capitalize()} unavailable: {exc}")

    try:
        if not collections:
            details = "; ".join(warnings) or "no feeds were requested"
            raise ValueError(f"No usable player-prop feed. {details}")
        players = merge_player_collections(*collections)
        if not players:
            raise ValueError("No supported full-game player prop markets were found")
        projections, logit_vig, calibration_pairs = make_projections(
            players,
            SCORING_PRESETS[args.scoring],
            fallback_logit_vig=args.fallback_logit_vig,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        output = {
            "generated_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "scoring": args.scoring,
            "source_mode": args.source,
            "feeds": loaded_feeds,
            "warnings": warnings,
            "one_way_logit_adjustment": round(logit_vig, 6),
            "calibration_pairs": calibration_pairs,
            "projections": [asdict(projection) for projection in projections],
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        print_projections(
            projections,
            args.scoring,
            logit_vig,
            calibration_pairs,
            args.include_td_only,
            loaded_feeds,
        )
    return 0




# ---------------------------------------------------------------------------
# Display-only reference for the lineup page
# ---------------------------------------------------------------------------
MARKET_DISPLAY_POSITIONS = ("QB", "RB", "WR", "TE")
# Qualities whose fantasy_points cover every core scoring component (td-estimate
# fills yards from a regression and is shown marked as an estimate).
COMPLETE_QUALITIES = ("good", "fair", "td-estimate")


def market_reference(normalize_name, normalize_team, timeout: float = 45.0,
                     scoring: str = "yahoo"):
    """Fetch both prop feeds and return (frame, notes).

    ``frame`` has Key, Team, Market_FP, Market_Quality: one row per player the
    books price this week, keyed the same way as the lineup roster. Either feed
    may fail; if both do the frame is empty and the notes say why. Games that
    have already kicked off are dropped, since their props are gone or live.
    """
    import pandas as pd

    collections, notes = [], []
    for feed, fetch, validate, parse in (
        ("bovada", fetch_bovada_payload, validate_bovada_payload, parse_bovada_payload),
        ("underdog", fetch_underdog_payload, validate_underdog_payload, parse_underdog_payload),
    ):
        try:
            payload = fetch(timeout=timeout)
            validate(payload)
            parsed = parse(payload)
            if not parsed:
                raise ValueError("no pregame NFL player props")
            collections.append(parsed)
            notes.append(f"{feed}: {len(parsed):,} player-market records")
        except Exception as exc:  # one feed down never blocks the other
            notes.append(f"{feed} unavailable ({type(exc).__name__}: {exc})")
    empty = pd.DataFrame(columns=["Key", "Team", "Market_FP", "Market_Quality", "Market_Props",
                                  "Market_Kickoff"])
    if not collections:
        return empty, notes
    merged = merge_player_collections(*collections)
    projections, _, _ = make_projections(
        merged, SCORING_PRESETS[scoring],
        fallback_logit_vig=DEFAULT_FALLBACK_LOGIT_VIG)
    by_player = {(normalize_name(m.name), normalize_team(m.team)): m for m in merged.values()}
    now = datetime.now(timezone.utc)
    rows = []
    for p in projections:
        start = getattr(p, "start_time_utc", None)
        try:
            if start and datetime.fromisoformat(str(start).replace("Z", "+00:00")) <= now:
                continue
        except ValueError:
            pass
        # A partial or unconverted TD-only sum is a subtotal, not a full-game
        # estimate: publish its props but no points, so it is never read
        # (or highlighted) as a whole projection.
        complete = p.quality in COMPLETE_QUALITIES
        rows.append({"Key": normalize_name(p.player), "Team": normalize_team(p.team),
                     "Market_FP": round(float(p.fantasy_points), 2) if complete else None,
                     # Kept so the lineup only attaches props for the game it is pricing.
                     "Market_Kickoff": start,
                     "Market_Quality": p.quality,
                     "Market_Props": props_summary(
                         by_player.get((normalize_name(p.player), normalize_team(p.team))),
                         p.stat_means, p.position)})
    if not rows:
        return empty, notes
    frame = pd.DataFrame(rows).drop_duplicates(["Key", "Team"], keep=False)
    return frame, notes


# Compact labels, in reading order, for the props line on the lineup page.
_PROP_ORDER = (
    ("passing_yards", "Pa"), ("passing_touchdowns", "PaTD"), ("interceptions", "INT"),
    ("rushing_yards", "Ru"), ("receptions", "Rec"), ("receiving_yards", "RecYd"),
)


def main_line(market) -> Optional[float]:
    """The book's posted main line (e.g. 56.5), or None if only alternates exist.

    The engine stores "over 56.5" as P[X >= 57], so the posted line is the
    threshold minus one half. Several books' main lines are averaged.
    """
    if market is None or not market.anchors:
        return None
    lines = [a.threshold - 0.5 for a in market.anchors]
    return round(sum(lines) / len(lines) * 2) / 2


def props_summary(player, stat_means, position=None) -> str:
    """One compact line of a player's props, e.g. "Ru 56.5 · Rec 2.5 · RecYd 12.5 · TD 34%".

    Posted main lines are shown as-is; a stat priced only through alternate
    lines shows the fitted mean with a "~". The anytime-TD chance is the
    de-vigged P[TD >= 1] implied by the fitted touchdown rate.
    """
    parts = []
    for stat, label in _PROP_ORDER:
        if stat not in (stat_means or {}):
            continue
        line = main_line(player.stats.get(stat)) if player is not None else None
        if line is not None:
            parts.append(f"{label} {line:g}")
        else:
            mean = stat_means[stat]
            parts.append(f"{label} ~{mean:.1f}" if stat in ("receptions", "passing_touchdowns",
                                                            "interceptions") else f"{label} ~{mean:.0f}")
    rate = (stat_means or {}).get("any_touchdowns")
    if rate is not None:
        parts.append(f"TD {round(100 * (1 - math.exp(-rate)))}%")
    return " \u00b7 ".join(parts)
