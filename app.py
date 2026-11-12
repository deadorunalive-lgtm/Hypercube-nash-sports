import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_data import current_season_start_year, load_team_data

# Page config
st.set_page_config(
    page_title="HyperCube-Nash Sports Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header { text-align: center; color: #666; margin-bottom: 2rem; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; border-left: 4px solid #1f77b4; }
    .win-prob { font-size: 2.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">HyperCube-Nash Sports Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">NBA season statistics and win-probability comparisons</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    season_start = current_season_start_year()
    seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(season_start, season_start - 6, -1)]
    season = st.selectbox("NBA Season", seasons)
    model_mode = st.radio(
        "Model Mode",
        ["Practical Baseline (Elo + Stats)", "HyperCube-Nash (Conceptual)"],
        help="The practical model uses an app-estimated Elo rating and official NBA season statistics. HyperCube-Nash is conceptual."
    )
    home_advantage = st.slider("Home Advantage (Elo points)", 0, 150, 80)
    st.markdown("---")
    st.caption("Team statistics are fetched from the NBA Stats API and cached for one hour.")

try:
    with st.spinner(f"Loading {season} NBA team statistics…"):
        teams_df = load_team_data.clear() if False else load_team_data(season)
except Exception as exc:
    st.error(f"Could not load real NBA statistics: {exc}")
    st.info("Check your connection and try again. NBA Stats may temporarily rate-limit requests; retry in a few minutes.")
    st.stop()

st.caption(f"Source: NBA Stats API · {season} regular season · Data may be incomplete early in the season.")

# Main content
col1, col2 = st.columns(2)
with col1:
    st.subheader("Select Teams")
    team_names = teams_df["Team"].tolist()
    team_a = st.selectbox("Team A (Home)", team_names, index=0)
    default_b = min(1, len(team_names) - 1)
    team_b = st.selectbox("Team B (Away)", team_names, index=default_b)

# Get team stats
def get_team_stats(name):
    return teams_df.loc[teams_df["Team"] == name].iloc[0]

stats_a = get_team_stats(team_a)
stats_b = get_team_stats(team_b)

with col2:
    st.subheader("Season Snapshot")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(team_a, f"{int(stats_a['Wins'])}-{int(stats_a['Losses'])}", f"{stats_a['Win_Pct']:.1%} win%")
        st.caption(f"Net Rating: {stats_a['Net_Rating']:+.1f}")
    with c2:
        st.metric(team_b, f"{int(stats_b['Wins'])}-{int(stats_b['Losses'])}", f"{stats_b['Win_Pct']:.1%} win%")
        st.caption(f"Net Rating: {stats_b['Net_Rating']:+.1f}")

st.markdown("---")

# Prediction logic
def elo_win_prob(elo_a, elo_b, home_adv=0):
    """Standard Elo win probability."""
    diff = (elo_a + home_adv) - elo_b
    return 1 / (1 + 10 ** (-diff / 400))


def stats_win_prob(stats_a, stats_b, home_adv=0):
    """Logistic model based on official NBA net rating and win percentage."""
    net_diff = stats_a["Net_Rating"] - stats_b["Net_Rating"]
    win_diff = stats_a["Win_Pct"] - stats_b["Win_Pct"]
    home_boost = home_adv / 100 * 2.5
    score = 0.08 * net_diff + 3.5 * win_diff + home_boost
    return 1 / (1 + np.exp(-score))

if model_mode == "Practical Baseline (Elo + Stats)":
    elo_prob = elo_win_prob(stats_a["Elo"], stats_b["Elo"], home_advantage)
    stats_prob = stats_win_prob(stats_a, stats_b, home_advantage)
    win_prob_a = 0.6 * elo_prob + 0.4 * stats_prob
    method = "Estimated Elo + NBA Net Rating Blend"
else:
    base = stats_win_prob(stats_a, stats_b, home_advantage)
    strength_gap = abs(stats_a["Net_Rating"] - stats_b["Net_Rating"])
    attractor_pull = min(0.08, strength_gap * 0.005)
    if stats_a["Net_Rating"] > stats_b["Net_Rating"]:
        win_prob_a = min(0.92, base + attractor_pull)
    else:
        win_prob_a = max(0.08, base - attractor_pull)
    method = "HyperCube-Nash (Attractor-adjusted)"

win_prob_b = 1 - win_prob_a
expected_margin = (stats_a["Point_Diff"] - stats_b["Point_Diff"]) * 0.7 + (home_advantage / 20)

# Display results
st.subheader("Prediction Results")
res1, res2, res3 = st.columns(3)
with res1:
    st.markdown(f"**{team_a}**")
    color = "#2ecc71" if win_prob_a > 0.5 else "#e74c3c"
    st.markdown(f'<p class="win-prob" style="color:{color}">{win_prob_a:.1%}</p>', unsafe_allow_html=True)
    st.caption("Win Probability")
with res2:
    st.markdown("**Expected Margin**")
    st.markdown(f'<p class="win-prob">{expected_margin:+.1f}</p>', unsafe_allow_html=True)
    st.caption(f"points for {team_a}")
with res3:
    st.markdown(f"**{team_b}**")
    color_b = "#2ecc71" if win_prob_b > 0.5 else "#e74c3c"
    st.markdown(f'<p class="win-prob" style="color:{color_b}">{win_prob_b:.1%}</p>', unsafe_allow_html=True)
    st.caption("Win Probability")

st.info(f"**Model:** {method}  |  **Home Advantage:** {home_advantage} Elo points")
st.markdown("---")
st.subheader("Analysis Visualizations")
tab1, tab2, tab3 = st.tabs(["Probability Breakdown", "Season Stats Comparison", "Dynamical View"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[win_prob_a, win_prob_b], y=[team_a, team_b], orientation="h",
        marker_color=["#2ecc71" if win_prob_a > 0.5 else "#e74c3c", "#2ecc71" if win_prob_b > 0.5 else "#e74c3c"],
        text=[f"{win_prob_a:.1%}", f"{win_prob_b:.1%}"], textposition="auto"
    ))
    fig.update_layout(title="Win Probability", xaxis_title="Probability", height=300, showlegend=False, xaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    categories = ["Win %", "Net Rating", "Off Rating", "Def Rating", "Pace"]
    def normalize(val, col):
        mn, mx = teams_df[col].min(), teams_df[col].max()
        return (val - mn) / (mx - mn) if mx != mn else 0.5
    vals_a = [normalize(stats_a["Win_Pct"], "Win_Pct"), normalize(stats_a["Net_Rating"], "Net_Rating"), normalize(stats_a["Off_Rating"], "Off_Rating"), 1 - normalize(stats_a["Def_Rating"], "Def_Rating"), normalize(stats_a["Pace"], "Pace")]
    vals_b = [normalize(stats_b["Win_Pct"], "Win_Pct"), normalize(stats_b["Net_Rating"], "Net_Rating"), normalize(stats_b["Off_Rating"], "Off_Rating"), 1 - normalize(stats_b["Def_Rating"], "Def_Rating"), normalize(stats_b["Pace"], "Pace")]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]], theta=categories + [categories[0]], fill="toself", name=team_a))
    fig.add_trace(go.Scatterpolar(r=vals_b + [vals_b[0]], theta=categories + [categories[0]], fill="toself", name=team_b))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=450, title="Normalized Season Strength Profile")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""
    **Conceptual Dynamical Systems View**

    This visualization is illustrative, not an NBA statistic or a validated forecasting method.
    The HyperCube-Nash mode is a conceptual attractor adjustment, not a calibrated model.
    """)
    x = np.linspace(0, 1, 100)
    if stats_a["Net_Rating"] > stats_b["Net_Rating"]:
        attractor, label = win_prob_a, f"{team_a} attractor"
    else:
        attractor, label = win_prob_b, f"{team_b} attractor"
    basin = np.exp(-((x - attractor) ** 2) / 0.02)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=basin, fill="tozeroy", name="Illustrative basin", line=dict(color="#1f77b4")))
    fig.add_vline(x=attractor, line_dash="dash", line_color="red", annotation_text=label)
    fig.update_layout(title="Simplified Attractor Basin (Conceptual)", xaxis_title="Win Probability Space", yaxis_title="Attraction Strength", height=350)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Full Season Stats Comparison"):
    compare = pd.DataFrame({
        "Metric": ["Games Played", "Record", "Win %", "PPG", "OPP PPG", "Point Diff", "Net Rating", "Off Rating", "Def Rating", "Pace", "Estimated Elo"],
        team_a: [
            f"{int(stats_a['GP'])}", f"{int(stats_a['Wins'])}-{int(stats_a['Losses'])}", f"{stats_a['Win_Pct']:.1%}",
            f"{stats_a['PPG']:.1f}", f"{stats_a['OPP_PPG']:.1f}", f"{stats_a['Point_Diff']:+.1f}",
            f"{stats_a['Net_Rating']:+.1f}", f"{stats_a['Off_Rating']:.1f}", f"{stats_a['Def_Rating']:.1f}",
            f"{stats_a['Pace']:.1f}", f"{stats_a['Elo']:.0f}"
        ],
        team_b: [
            f"{int(stats_b['GP'])}", f"{int(stats_b['Wins'])}-{int(stats_b['Losses'])}", f"{stats_b['Win_Pct']:.1%}",
            f"{stats_b['PPG']:.1f}", f"{stats_b['OPP_PPG']:.1f}", f"{stats_b['Point_Diff']:+.1f}",
            f"{stats_b['Net_Rating']:+.1f}", f"{stats_b['Off_Rating']:.1f}", f"{stats_b['Def_Rating']:.1f}",
            f"{stats_b['Pace']:.1f}", f"{stats_b['Elo']:.0f}"
        ]
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)
    st.caption("NBA Stats API fields are official season aggregates. Estimated Elo is calculated by this app; it is not published by the NBA.")

st.markdown("---")
st.caption("Live season stats from the NBA Stats API. Statistics update as NBA games are recorded; data is cached for one hour. Predictions are estimates, not betting advice.")
