"""
Fetch live game state from the NBA API and translate it into model-ready features.

Uses the stats.nba.com ScoreboardV3 endpoint, which is not blocked by the CDN's
bot-detection layer and updates score, period and clock while games are in
progress. (Play-by-play is NOT an option for live games: PlayByPlayV3 returns no
actions until a game is final.) This module handles fetch + parse only; model
inference stays in predict.py / the app.
"""

from __future__ import annotations

import re
from datetime import date

from nba_api.stats.endpoints import scoreboardv3

# Game status codes returned by ScoreboardV3.
STATUS_PRE = 1
STATUS_LIVE = 2
STATUS_FINAL = 3


def today_games() -> list[dict]:
    """All of today's games from the NBA scoreboard.

    Returns dicts with: game_id, home, away, status (1/2/3 = pre/live/final),
    period, game_clock (raw PT...S string), home_score, away_score.
    """
    # league_id "00" = regular season + playoffs only. Pre-season ("01") is
    # excluded: experimental lineups and reduced starter minutes add too much
    # noise for a model trained on regular season data.
    sb = scoreboardv3.ScoreboardV3(
        game_date=date.today().strftime("%Y-%m-%d"), league_id="00"
    )
    out = []
    for g in sb.get_dict()["scoreboard"]["games"]:
        out.append({
            "game_id":    g["gameId"],
            "home":       g["homeTeam"]["teamTricode"],
            "away":       g["awayTeam"]["teamTricode"],
            "status":     g["gameStatus"],
            "period":     g["period"],
            "game_clock": g.get("gameClock", ""),
            "home_score": g["homeTeam"]["score"],
            "away_score": g["awayTeam"]["score"],
        })
    return out


def parse_clock(clock: str) -> float:
    """Return seconds remaining in the current period from an NBA clock string.

    Format: "PT05M42.00S" (ISO 8601 duration). Returns 0.0 for an empty string
    (between periods / pre-game).

    Examples:
        "PT05M42.00S" -> 342.0
        "PT00M00.00S" -> 0.0
        ""            -> 0.0
    """
    if not clock:
        return 0.0
    m = re.search(r"PT(\d+)M([\d.]+)S", clock)
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def scoreboard_game_state(game: dict) -> dict:
    """Game state for one today_games() entry, ready for game_state_seconds().

    Keys: period, clock_seconds (left in this period), score_diff (home - away),
    is_overtime. Between periods the clock is empty, which parses to 0.0 - the
    end of the period just played, which is where the game actually stands.
    """
    period = max(1, int(game.get("period") or 1))
    home_score = int(game.get("home_score") or 0)
    away_score = int(game.get("away_score") or 0)
    return {
        "period":        period,
        "clock_seconds": parse_clock(game.get("game_clock", "")),
        "score_diff":    home_score - away_score,
        "is_overtime":   int(period > 4),
    }
