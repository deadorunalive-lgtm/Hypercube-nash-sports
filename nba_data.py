from datetime import date
import requests
import pandas as pd
import streamlit as st

NBA_STATS_URL = "https://stats.nba.com/stats/leaguedashteamstats"
NBA_TEAMS = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",
    1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",
    1610612752: "New York Knicks",
    1610612753: "Orlando Magic",
    1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612763: "Memphis Grizzlies",
    1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",
    1610612766: "Charlotte Hornets",
}


def current_season_start_year(today=None):
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def _fetch_team_stats(season, measure_type):
    params = {
        "College": "", "Conference": "", "Country": "", "DateFrom": "",
        "DateTo": "", "Division": "", "DraftPick": "", "DraftYear": "",
        " GameScope": "", "GameSegment": "", "Height": "", "LastNGames": 0,
        "LeagueID": "00", "Location": "", "MeasureType": measure_type,
        "Month": 0, "OpponentTeamID": 0, "Outcome": "", "PORound": 0,
        "PaceAdjust": "N", "PerMode": "PerGame", "PlayerExperience": "",
        "PlayerPosition": "", "PlusMinus": "N", "Rank": "N",
        "Season": season, "SeasonSegment": "", "SeasonType": "Regular Season",
        "ShotClockRange": "", "StarterBench": "", "TeamID": 0,
        "TwoWay": 0, "VsConference": "", "VsDivision": "",
    }
    # The NBA endpoint expects GameScope (without a leading space).
    params.pop(" GameScope")
    response = requests.get(
        NBA_STATS_URL,
        params=params,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
            "Accept": "application/json, text/plain, */*",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    result_sets = payload.get("resultSets") or payload.get("resultSet")
    if not result_sets:
        raise ValueError(f"NBA stats returned no {measure_type.lower()} team data for {season}.")
    result = result_sets[0] if isinstance(result_sets, list) else result_sets
    return pd.DataFrame(result.get("rowSet", []), columns=result.get("headers", []))


@st.cache_data(ttl=3600, show_spinner=False)
def load_team_data(season):
    """Fetch official NBA regular-season team statistics; cache for one hour."""
    base = _fetch_team_stats(season, "Base")
    advanced = _fetch_team_stats(season, "Advanced")
    if base.empty or advanced.empty:
        raise ValueError(f"NBA stats are not available yet for the {season} regular season.")

    required_base = {"TEAM_ID", "GP", "W", "L", "W_PCT", "PTS", "PLUS_MINUS"}
    required_advanced = {"TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"}
    if not required_base.issubset(base.columns) or not required_advanced.issubset(advanced.columns):
        raise ValueError("The NBA stats response format changed; please try again later.")

    stats = base.merge(
        advanced[["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]],
        on="TEAM_ID", how="inner", validate="one_to_one",
    )
    stats = stats[stats["TEAM_ID"].isin(NBA_TEAMS)].copy()
    stats = stats[stats["GP"] > 0]
    if stats.empty:
        raise ValueError(f"No games have been recorded for the {season} regular season yet.")

    stats["Team"] = stats["TEAM_ID"].map(NBA_TEAMS)
    stats["Wins"] = stats["W"]
    stats["Losses"] = stats["L"]
    stats["Win_Pct"] = stats["W_PCT"]
    stats["PPG"] = stats["PTS"]
    # PLUS_MINUS is the official per-game point differential, so derive opponent PPG.
    stats["Point_Diff"] = stats["PLUS_MINUS"]
    stats["OPP_PPG"] = stats["PPG"] - stats["Point_Diff"]
    # Elo is an app-side estimate, not an NBA-published statistic.
    stats["Elo"] = 1500 + (stats["Win_Pct"] - 0.5) * 400 + stats["NET_RATING"] * 12
    stats = stats.rename(columns={
        "NET_RATING": "Net_Rating", "OFF_RATING": "Off_Rating",
        "DEF_RATING": "Def_Rating",
    })
    return stats.sort_values("Team").reset_index(drop=True)
