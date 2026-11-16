"""
Knicks Win Probability - interactive showcase.

Three views:
  - Overview: one team's odds against every other team, at a glance.
  - Matchup Drill-Down: P(home win) for any two teams at any game state.
  - Game Replay: the win-probability curve through any historical game.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from live_feed import (  # noqa: E402
    STATUS_LIVE,
    scoreboard_game_state,
    today_games,
)
from predict import (  # noqa: E402
    MatchupPredictor,
    endgame_certainty,
    features_path,
    game_state_seconds,
    load_model,
)
from serving_data import serving_file, serving_model  # noqa: E402

# Serving artifacts come from the release-or-snapshot loader - no DuckDB, no
# pipeline. The committed snapshot below is the offline fallback. The caches use
# a TTL so the live app picks up the daily release refresh without a redeploy.
CACHE_TTL = 6 * 3600
MODEL_PATH = PROJECT_ROOT / "models" / "win_probability.json"
GAMES_PATH = PROJECT_ROOT / "data" / "serving" / "games.parquet"
REPLAY_PATH = PROJECT_ROOT / "data" / "serving" / "replay.parquet"
TEAMS_PATH = PROJECT_ROOT / "data" / "serving" / "teams.parquet"


@st.cache_data(ttl=CACHE_TTL)
def teams_dim() -> pd.DataFrame:
    """The conformed team dimension (tricode, canonical_tricode, full_name,
    is_current), exported from dbt's dim_teams - the single source of truth."""
    return pd.read_parquet(serving_file("teams.parquet", TEAMS_PATH))


@st.cache_data(ttl=CACHE_TTL)
def _name_map() -> dict[str, str]:
    df = teams_dim()
    return dict(zip(df["tricode"], df["full_name"]))


def name(tricode: str) -> str:
    """Full franchise name for any tricode, current or historical."""
    return _name_map().get(tricode, tricode)


@st.cache_resource(ttl=CACHE_TTL)
def get_model():
    return load_model(serving_model(MODEL_PATH, features_path(MODEL_PATH)))


@st.cache_resource(ttl=CACHE_TTL)
def get_predictor() -> MatchupPredictor:
    return MatchupPredictor()


@st.cache_data(ttl=CACHE_TTL)
def all_games() -> pd.DataFrame:
    return pd.read_parquet(serving_file("games.parquet", GAMES_PATH)).sort_values(
        "game_date", ascending=False)


@st.cache_data(ttl=CACHE_TTL)
def all_replay() -> pd.DataFrame:
    return pd.read_parquet(serving_file("replay.parquet", REPLAY_PATH))


def list_games(team: str | None) -> pd.DataFrame:
    games = all_games()
    if team:
        games = games[(games["home_abbr"] == team) | (games["away_abbr"] == team)]
    return games


@st.cache_data
def game_curve(game_id: str) -> pd.DataFrame:
    df = all_replay()
    df = df[df["game_id"] == game_id].sort_values(["period", "action_number"]).copy()
    model, features = get_model()
    df["p_home"] = model.predict_proba(df[features])[:, 1]
    # Snap decided end-of-period moments to certainty (overrides the model).
    decided = [endgame_certainty(s, d) for s, d in zip(df["seconds_remaining"], df["score_diff"])]
    df["p_home"] = [c if c is not None else p for c, p in zip(decided, df["p_home"])]
    df["elapsed"] = [elapsed_seconds(p, s)
                     for p, s in zip(df["period"], df["seconds_remaining"])]
    df["clock"] = [game_clock(p, s)
                   for p, s in zip(df["period"], df["seconds_remaining"])]
    # The feed's action order has occasional clock anomalies (an event stamped
    # earlier in the game listed after a later one). Draw the curve in true time
    # order so the line never doubles back; action_number breaks ties at a whistle.
    df = df.sort_values(["elapsed", "action_number"]).reset_index(drop=True)
    return df


def period_name(period: int) -> str:
    """Q1-Q4 in regulation; OT, 2OT, ... beyond."""
    if period <= 4:
        return f"Q{period}"
    ot = period - 4
    return "OT" if ot == 1 else f"{ot}OT"


def game_clock(period: int, seconds_remaining: float) -> str:
    """Human-readable game clock for hover, e.g. "Q3 5:42".

    seconds_remaining is left in the whole game during regulation, so we strip
    out the later quarters to get the clock within this period; in OT it already
    is the period's own clock.
    """
    in_period = seconds_remaining - (4 - period) * 720 if period <= 4 else seconds_remaining
    mins, secs = divmod(int(in_period), 60)
    return f"{period_name(period)} {mins}:{secs:02d}"


def period_start_elapsed(period: int) -> float:
    """Elapsed seconds at the tip of a given period (where its tick sits)."""
    if period <= 4:
        return (period - 1) * 720
    return REGULATION_SECONDS + (period - 5) * OT_SECONDS


def quarter_ticks(periods: list[int]) -> tuple[list[float], list[str]]:
    """Tick (position, label) pairs at the start of each period in the game."""
    starts = sorted(set(periods))
    vals = [period_start_elapsed(p) for p in starts]
    return vals, [period_name(p) for p in starts]


@st.cache_data
def teams() -> list[str]:
    """Current franchises only (tricodes), for the pickers - excludes historical
    aliases so the dropdowns show today's 30 teams."""
    df = teams_dim()
    return sorted(df[df["is_current"]]["tricode"])


REGULATION_SECONDS = 2880  # 4 x 12-minute quarters
OT_SECONDS = 300           # each overtime period is 5 minutes


def elapsed_seconds(period: int, seconds_remaining: float) -> float:
    """Game time elapsed (counts up from 0), from the model's countdown clock.

    seconds_remaining counts DOWN: in regulation it's time left in the whole
    game (2880 -> 0); in OT it resets to that single 5-minute period's clock.
    We invert it so the replay x-axis runs left-to-right, with quarter
    boundaries landing on clean multiples of 720.
    """
    if period <= 4:
        return REGULATION_SECONDS - seconds_remaining
    return (REGULATION_SECONDS
            + (period - 5) * OT_SECONDS
            + (OT_SECONDS - seconds_remaining))


def field_matchups(predictor: MatchupPredictor, team: str, at_home: bool,
                   is_playoff: int = 0) -> pd.DataFrame:
    """One team's tip-off win probability against every other current team.

    Returns a DataFrame with columns "opponent" (tricode) and "p_win"
    (P(`team` wins)), one row per opposing team. The Team vs Field tab renders
    this as a sorted bar chart, so the row order here is the bar order.
    """
    opponents = [opp for opp in teams() if opp != team]
    odds = [
        {"opponent": opp, "p_win": predictor.win_probability(team, opp, is_playoff=is_playoff) if at_home
                                else 1 - predictor.win_probability(opp, team, is_playoff=is_playoff)}
        for opp in opponents
    ]
    return pd.DataFrame(odds)

# --- layout ---------------------------------------------------------------
st.set_page_config(page_title="NBA Win Probability", page_icon="🏀", layout="wide")
st.title("🏀 NBA Win Probability")

overview_tab, odds_tab, replay_tab, live_tab = st.tabs(
    ["Overview", "Matchup Drill-Down", "Game Replay", "Live"])

with odds_tab:
    st.caption("Pick two teams for the outright odds at tip-off. "
               "Expand the drill-down to set a specific moment in the game.")
    predictor = get_predictor()
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams(), format_func=name,
                        index=teams().index("NYK") if "NYK" in teams() else 0)
    away = c2.selectbox("Away team", teams(), format_func=name, index=0)

    is_playoff = int(st.toggle("Playoff game", key="odds_playoff"))

    # Game-state controls live in a collapsed drill-down, so the default view is
    # just the tip-off odds. Their defaults (Q1, full clock, tied) = game start.
    PERIODS = {"1st Quarter": 1, "2nd Quarter": 2, "3rd Quarter": 3,
               "4th Quarter": 4, "Overtime": 5}
    with st.expander("Drill down to a specific moment"):
        period_label = st.selectbox("Period", list(PERIODS))
        period = PERIODS[period_label]
        period_minutes = 12 if period <= 4 else 5  # OT periods are 5 minutes
        t1, t2 = st.columns(2)
        mins = t1.number_input("Minutes left", min_value=0, max_value=period_minutes,
                               value=period_minutes, step=1)
        secs = t2.number_input("Seconds left", min_value=0, max_value=59, value=0, step=1)
        # Clock as total seconds (minutes:seconds is base-60), capped at the period.
        clock_seconds = min(mins * 60 + secs, period_minutes * 60)
        margin = st.slider("Home margin (home score - away score)", -30, 30, 0)
    seconds_remaining, is_overtime = game_state_seconds(period, clock_seconds)

    if home == away:
        st.warning("Pick two different teams.")
    else:
        p = predictor.win_probability(
            home, away, seconds_remaining=seconds_remaining,
            score_diff=margin, is_overtime=is_overtime, is_playoff=is_playoff,
        )
        m1, m2 = st.columns(2)
        m1.metric(f"{name(home)} win", f"{p:.1%}")
        m2.metric(f"{name(away)} win", f"{1 - p:.1%}")

        at_tipoff = not is_overtime and seconds_remaining >= 2880 and margin == 0
        moment = ("at tip-off" if at_tipoff else
                  f"with {clock_seconds // 60}:{clock_seconds % 60:02d} left in "
                  f"{period_label.lower()}, home {margin:+d}")
        st.caption(
            f"Win probability {moment}.  Elo - {name(home)}: "
            f"{predictor.ratings[home]:.0f}, {name(away)}: {predictor.ratings[away]:.0f}"
        )

with overview_tab:
    st.caption("See how one team's tip-off odds stack up against the whole league, "
               "then pick an opponent to drill into the head-to-head.")
    predictor = get_predictor()
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    team = fc1.selectbox("Team", teams(), format_func=name, key="field_team",
                         index=teams().index("NYK") if "NYK" in teams() else 0)
    court = fc2.radio("Court", ["Home", "Away"], horizontal=True, key="field_court")
    is_playoff = int(fc3.toggle("Playoff game", key="field_playoff"))

    field = field_matchups(predictor, team, at_home=(court == "Home"),
                           is_playoff=is_playoff)

    SORT_OPTIONS = {
        "Team Name: A to Z":            ("opponent", True),
        "Team Name: Z to A":            ("opponent", False),
        "Win Probability: High to Low": ("p_win", False),
        "Win Probability: Low to High": ("p_win", True)
    }
    sort_label = st.selectbox("Sort by", list(SORT_OPTIONS), key="field_sort")
    sort_col, ascending = SORT_OPTIONS[sort_label]
    # Horizontal bar charts render bottom-to-top, so flip ascending to put the
    # "best" row (highest prob or A) at the top of the chart, not the bottom.
    field = field.sort_values(sort_col, ascending=not ascending).reset_index(drop=True)

    # Knicks orange where the team is favored, gray where it's the underdog -
    # the 50% line splits the field into wins and losses at a glance.
    colors = ["#F58426" if p >= 0.5 else "#9aa0a6" for p in field["p_win"]]
    fig = go.Figure(go.Bar(
        x=field["p_win"], y=[name(o) for o in field["opponent"]],
        orientation="h", marker_color=colors,
        text=[f"{p:.0%}" for p in field["p_win"]], textposition="auto",
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
    fig.update_layout(
        xaxis=dict(title=f"Percent Chance of {name(team)} Winning ({court})",
                   range=[0, 1], tickformat=".0%"),
        height=max(440, 20 * len(field)), margin=dict(t=30, l=10),
    )
    event = st.plotly_chart(fig, width="stretch",
                            on_select="rerun", selection_mode="points")

    # Drill-down appears only after the user clicks a bar.
    if event.selection.points:
        selected_name = event.selection.points[0]["y"]
        name_to_tricode = {name(o): o for o in field["opponent"]}
        opp = name_to_tricode.get(selected_name)
        if opp:
            p = field.set_index("opponent").loc[opp, "p_win"]
            st.divider()
            d1, d2 = st.columns(2)
            d1.metric(f"{name(team)} win", f"{p:.1%}")
            d2.metric(f"{name(opp)} win", f"{1 - p:.1%}")
            game_context = ("Playoff" if is_playoff else "Regular season") + f", {court.lower()}"
            st.caption(f"{game_context}.  Elo - {name(team)}: {predictor.ratings[team]:.0f}, "
                       f"{name(opp)}: {predictor.ratings[opp]:.0f}")

with replay_tab:
    st.caption("Drill into a single game: the win-probability curve as it played out "
               "(the same view a live game would stream into).")
    team = st.selectbox("Filter by team", ["(all)"] + teams(),
                        format_func=lambda t: "All teams" if t == "(all)" else name(t))
    games = list_games(None if team == "(all)" else team)

    labels = {
        row.game_id: f"{row.game_date}   {name(row.away_abbr)} @ {name(row.home_abbr)}   "
                     f"({int(row.away_pts)}-{int(row.home_pts)})"
        for row in games.itertuples(index=False)
    }
    game_id = st.selectbox("Game", list(labels), format_func=lambda g: labels[g])

    if game_id:
        df = game_curve(game_id)
    if game_id and df.empty:
        st.info("No play-by-play for this game yet - it will appear after the next "
                "data refresh.")
    elif game_id:
        g = games[games["game_id"] == game_id].iloc[0]
        home, away = g.home_abbr, g.away_abbr
        home_won = g.home_pts > g.away_pts

        # Default to the filtered team's perspective; fall back to home.
        default_idx = 1 if team == away else 0
        perspective = st.radio(
            "Show odds for",
            [name(home), name(away)],
            index=default_idx,
            horizontal=True,
            key=f"replay_perspective_{game_id}",
        )
        show_home = perspective == name(home)
        viewed_team = home if show_home else away
        p_curve = df["p_home"] if show_home else 1 - df["p_home"]

        tickvals, ticktext = quarter_ticks(df["period"].tolist())
        # Live scoreboard at each moment, e.g. "NYK 58 - 52 SAS". Always shown
        # home-then-away so it reads the same regardless of the perspective toggle.
        scoreboard = [f"{home} {int(h)} - {int(a)} {away}"
                      for h, a in zip(df["score_home"], df["score_away"])]
        hover = list(zip(df["clock"], scoreboard))
        fig = go.Figure()
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
        # Faint vertical line at each quarter boundary, so the periods read as
        # distinct segments rather than one continuous sweep.
        for x in tickvals[1:]:
            fig.add_vline(x=x, line_dash="dot", line_color="#e0e0e0")
        fig.add_trace(go.Scatter(
            x=df["elapsed"], y=p_curve, mode="lines",
            line=dict(color="#F58426", width=2), name=f"P({viewed_team} win)",
            customdata=hover,
            hovertemplate=(f"%{{customdata[0]}}<br>%{{customdata[1]}}<br>"
                           f"{name(viewed_team)} win: %{{y:.0%}}<extra></extra>"),
        ))
        fig.update_layout(
            yaxis=dict(title=f"Percent Chance of {name(viewed_team)} Winning",
                       range=[0, 1], tickformat=".0%"),
            xaxis=dict(title="Game progression", tickvals=tickvals, ticktext=ticktext),
            height=440, margin=dict(t=30),
        )
        st.plotly_chart(fig, width="stretch")

        winner = home if home_won else away
        st.markdown(
            f"**Final:** {name(away)} {int(g.away_pts)} - {int(g.home_pts)} {name(home)}  →  "
            f"**{name(winner)} won.**  Pre-game model: P({name(viewed_team)} win) = {p_curve.iloc[0]:.0%}"
        )

@st.cache_resource
def live_history() -> tuple[dict[str, list[dict]], threading.Lock]:
    """Scoreboard readings per live game, shared across every viewer session.

    Live play-by-play isn't available (PlayByPlayV3 is empty until a game is
    final), so the live curve is built from successive scoreboard snapshots.
    Keeping them server-side means the curve covers the whole time anyone has
    had the app open, not just the current viewer's session.
    """
    return {}, threading.Lock()


def live_features(state: dict, home: str, away: str) -> dict:
    """One model-ready feature row for a live game state."""
    predictor = get_predictor()
    rating_diff = predictor.ratings.get(home, 0) - predictor.ratings.get(away, 0)
    home_rapm, home_box = predictor.roster_value.get(home, (None, None))
    away_rapm, away_box = predictor.roster_value.get(away, (None, None))
    secs, is_ot = game_state_seconds(state["period"], state["clock_seconds"])
    return {
        "seconds_remaining":     secs,
        "score_diff":            state["score_diff"],
        "is_overtime":           is_ot,
        "is_playoff":            0,
        "rating_diff":           rating_diff,
        "roster_value_diff":     (home_rapm - away_rapm
                                  if home_rapm is not None and away_rapm is not None else 0.0),
        "roster_value_box_diff": (home_box - away_box
                                  if home_box is not None and away_box is not None else 0.0),
        "period":                state["period"],
    }


def live_p_home(row: dict) -> float:
    model, features = get_model()
    decided = endgame_certainty(row["seconds_remaining"], row["score_diff"])
    if decided is not None:
        return decided
    return float(model.predict_proba(pd.DataFrame([row])[features])[:, 1][0])


def record_live_reading(game: dict) -> list[dict]:
    """Append this scoreboard snapshot to the game's history (if it changed)
    and return the history. The curve is seeded with the pre-game number at
    tip-off so it always starts from the model's prior."""
    home, away = game["home"], game["away"]
    row = live_features(scoreboard_game_state(game), home, away)
    point = {
        "period":     row["period"],
        "elapsed":    elapsed_seconds(row["period"], row["seconds_remaining"]),
        "score_home": int(game["home_score"] or 0),
        "score_away": int(game["away_score"] or 0),
        "p_home":     live_p_home(row),
    }
    history, lock = live_history()
    with lock:
        points = history.setdefault(game["game_id"], [])
        if not points:
            tip = live_features({"period": 1, "clock_seconds": 720.0, "score_diff": 0},
                                home, away)
            points.append({"period": 1, "elapsed": 0.0, "score_home": 0,
                           "score_away": 0, "p_home": live_p_home(tip)})
        last = points[-1]
        if (point["elapsed"], point["score_home"], point["score_away"]) != (
                last["elapsed"], last["score_home"], last["score_away"]):
            points.append(point)
        return list(points)


def prune_live_history(live_ids: set[str]) -> None:
    """Drop history for games that are no longer in progress."""
    history, lock = live_history()
    with lock:
        for gid in [g for g in history if g not in live_ids]:
            del history[gid]


with live_tab:
    st.caption("Win probability updating in real time for any game currently in progress. "
               "Refreshes every 30 seconds from the live scoreboard; the curve builds "
               "from when the app was first opened during the game.")

    @st.fragment(run_every=30)
    def live_section() -> None:
        try:
            games = today_games()
        except Exception as exc:
            st.warning(f"Could not reach the NBA scoreboard: {exc}")
            return

        in_progress = [g for g in games if g["status"] == STATUS_LIVE]
        prune_live_history({g["game_id"] for g in in_progress})
        if not in_progress:
            other = [g for g in games if g["status"] != STATUS_LIVE]
            if other:
                labels = [f"{name(g['away'])} @ {name(g['home'])}" for g in other]
                st.info("No games in progress right now.  Today's other games: "
                        + ",  ".join(labels))
            else:
                st.info("No NBA games scheduled today.")
            return

        # Record every live game each tick, not just the one on screen, so
        # switching games shows a curve that has been building all along.
        histories = {g["game_id"]: record_live_reading(g) for g in in_progress}

        game_labels = {
            g["game_id"]: f"{name(g['away'])} @ {name(g['home'])}"
            for g in in_progress
        }
        selected_id = st.selectbox(
            "Game", list(game_labels), format_func=lambda gid: game_labels[gid],
            key="live_game_picker",
        )
        selected = next(g for g in in_progress if g["game_id"] == selected_id)
        home, away = selected["home"], selected["away"]
        df = pd.DataFrame(histories[selected_id])

        perspective = st.radio(
            "Show odds for", [name(home), name(away)],
            horizontal=True, key="live_perspective",
        )
        show_home = perspective == name(home)
        viewed = home if show_home else away
        p_curve = df["p_home"] if show_home else 1 - df["p_home"]
        latest_p = float(p_curve.iloc[-1])

        m1, m2 = st.columns(2)
        m1.metric(f"{name(home)} score", selected["home_score"])
        m2.metric(f"{name(away)} score", selected["away_score"])
        st.metric(f"P({name(viewed)} win)", f"{latest_p:.1%}")

        # Label every period up to the current one, even ones the curve skipped
        # (the app may have been opened mid-game).
        tickvals, ticktext = quarter_ticks(list(range(1, int(df["period"].max()) + 1)))
        scoreboard =[f"{home} {h} - {a} {away}"
                      for h, a in zip(df["score_home"], df["score_away"])]
        fig = go.Figure()
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
        for x in tickvals[1:]:
            fig.add_vline(x=x, line_dash="dot", line_color="#e0e0e0")
        fig.add_trace(go.Scatter(
            x=df["elapsed"], y=p_curve, mode="lines+markers",
            line=dict(color="#F58426", width=2), marker=dict(size=4),
            name=f"P({viewed} win)", customdata=scoreboard,
            hovertemplate=(f"%{{customdata}}<br>{name(viewed)} win: "
                           f"%{{y:.0%}}<extra></extra>"),
        ))
        fig.update_layout(
            yaxis=dict(title=f"Percent Chance of {name(viewed)} Winning",
                       range=[0, 1], tickformat=".0%"),
            xaxis=dict(title="Game progression", tickvals=tickvals, ticktext=ticktext),
            height=440, margin=dict(t=30),
        )
        st.plotly_chart(fig, width="stretch")

    live_section()
