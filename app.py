import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .win-prob {
        font-size: 2.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">HyperCube-Nash Sports Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Season-stats based win probability engine with dynamical systems & game-theoretic foundations</p>', unsafe_allow_html=True)

# Mock NBA season data (realistic 2025-26 style stats)
@st.cache_data
def load_team_data():
    data = {
        "Team": [
            "Boston Celtics", "Denver Nuggets", "Milwaukee Bucks", "Philadelphia 76ers",
            "Phoenix Suns", "Golden State Warriors", "Miami Heat", "Los Angeles Lakers",
            "Dallas Mavericks", "New York Knicks", "Cleveland Cavaliers", "Minnesota Timberwolves",
            "Oklahoma City Thunder", "Sacramento Kings", "Indiana Pacers", "Orlando Magic"
        ],
        "Wins": [48, 46, 44, 42, 41, 40, 39, 38, 37, 36, 45, 43, 50, 35, 40, 41],
        "Losses": [22, 24, 26, 28, 29, 30, 31, 32, 33, 34, 25, 27, 20, 35, 30, 29],
        "PPG": [118.2, 116.8, 119.5, 115.1, 117.3, 116.9, 112.4, 114.7, 118.9, 113.8, 117.1, 115.6, 120.4, 116.2, 121.1, 112.9],
        "OPP_PPG": [109.8, 111.2, 112.4, 110.5, 113.1, 112.7, 108.9, 111.8, 115.3, 110.2, 108.4, 109.7, 107.8, 114.6, 116.2, 109.1],
        "Net_Rating": [8.4, 5.6, 7.1, 4.6, 4.2, 4.2, 3.5, 2.9, 3.6, 3.6, 8.7, 5.9, 12.6, 1.6, 4.9, 3.8],
        "Pace": [99.2, 98.1, 101.4, 97.8, 100.3, 101.8, 96.5, 99.7, 102.1, 97.4, 98.9, 99.5, 100.8, 101.2, 103.4, 96.8],
        "Off_Rating": [118.5, 116.2, 119.8, 114.9, 117.1, 116.5, 112.1, 114.4, 119.2, 113.5, 116.8, 115.3, 120.9, 116.0, 121.5, 112.6],
        "Def_Rating": [110.1, 110.6, 112.7, 110.3, 112.9, 112.3, 108.6, 111.5, 115.6, 109.9, 108.1, 109.4, 108.3, 114.4, 116.6, 108.8],
        "Elo": [1685, 1652, 1648, 1610, 1605, 1602, 1588, 1575, 1570, 1568, 1678, 1640, 1720, 1545, 1595, 1590]
    }
    df = pd.DataFrame(data)
    df["Win_Pct"] = df["Wins"] / (df["Wins"] + df["Losses"])
    df["Point_Diff"] = df["PPG"] - df["OPP_PPG"]
    return df

teams_df = load_team_data()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    league = st.selectbox("League", ["NBA (Demo)", "NFL (Coming)", "MLB (Coming)"])
    model_mode = st.radio(
        "Model Mode",
        ["Practical Baseline (Elo + Stats)", "HyperCube-Nash (Conceptual)"],
        help="Practical uses real statistical methods. HyperCube-Nash is the advanced dynamical/game-theoretic vision."
    )
    home_advantage = st.slider("Home Advantage (Elo points)", 0, 150, 80)
    st.markdown("---")
    st.caption("Built from dynamical systems + game theory notes. This demo uses realistic mock season data.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Select Teams")
    team_a = st.selectbox("Team A (Home)", teams_df["Team"].tolist(), index=0)
    team_b = st.selectbox("Team B (Away)", teams_df["Team"].tolist(), index=1)

# Get team stats
def get_team_stats(name):
    return teams_df[teams_df["Team"] == name].iloc[0]

stats_a = get_team_stats(team_a)
stats_b = get_team_stats(team_b)

with col2:
    st.subheader("Season Snapshot")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(team_a.split()[-1], f"{stats_a['Wins']}-{stats_a['Losses']}", f"{stats_a['Win_Pct']:.1%} win%")
        st.caption(f"Net Rating: {stats_a['Net_Rating']:+.1f}")
    with c2:
        st.metric(team_b.split()[-1], f"{stats_b['Wins']}-{stats_b['Losses']}", f"{stats_b['Win_Pct']:.1%} win%")
        st.caption(f"Net Rating: {stats_b['Net_Rating']:+.1f}")

st.markdown("---")

# Prediction logic
def elo_win_prob(elo_a, elo_b, home_adv=0):
    """Standard Elo win probability"""
    diff = (elo_a + home_adv) - elo_b
    return 1 / (1 + 10 ** (-diff / 400))

def stats_win_prob(stats_a, stats_b, home_adv=0):
    """Logistic model based on net rating and win% difference"""
    net_diff = stats_a["Net_Rating"] - stats_b["Net_Rating"]
    win_diff = stats_a["Win_Pct"] - stats_b["Win_Pct"]
    # Home advantage boost
    home_boost = home_adv / 100 * 2.5  # scale
    score = 0.08 * net_diff + 3.5 * win_diff + home_boost
    return 1 / (1 + np.exp(-score))

# Calculate probabilities
if model_mode == "Practical Baseline (Elo + Stats)":
    elo_prob = elo_win_prob(stats_a["Elo"], stats_b["Elo"], home_advantage)
    stats_prob = stats_win_prob(stats_a, stats_b, home_advantage)
    # Blend
    win_prob_a = 0.6 * elo_prob + 0.4 * stats_prob
    method = "Elo + Net Rating Blend"
else:
    # Conceptual HyperCube-Nash: slightly more variance + attractor adjustment
    base = stats_win_prob(stats_a, stats_b, home_advantage)
    # Simulate "basin of attraction" pull toward stronger team
    strength_gap = abs(stats_a["Net_Rating"] - stats_b["Net_Rating"])
    attractor_pull = min(0.08, strength_gap * 0.005)
    if stats_a["Net_Rating"] > stats_b["Net_Rating"]:
        win_prob_a = min(0.92, base + attractor_pull)
    else:
        win_prob_a = max(0.08, base - attractor_pull)
    method = "HyperCube-Nash (Attractor-adjusted)"

win_prob_b = 1 - win_prob_a

# Expected margin (simple)
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
    st.caption(f"points for {team_a.split()[-1]}")

with res3:
    st.markdown(f"**{team_b}**")
    color_b = "#2ecc71" if win_prob_b > 0.5 else "#e74c3c"
    st.markdown(f'<p class="win-prob" style="color:{color_b}">{win_prob_b:.1%}</p>', unsafe_allow_html=True)
    st.caption("Win Probability")

st.info(f"**Model:** {method}  |  **Home Advantage:** {home_advantage} Elo points")

# Visualizations
st.markdown("---")
st.subheader("Analysis Visualizations")

tab1, tab2, tab3 = st.tabs(["Probability Breakdown", "Season Stats Comparison", "Dynamical View"])

with tab1:
    # Gauge-style bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[win_prob_a, win_prob_b],
        y=[team_a.split()[-1], team_b.split()[-1]],
        orientation='h',
        marker_color=['#2ecc71' if win_prob_a > 0.5 else '#e74c3c', 
                      '#2ecc71' if win_prob_b > 0.5 else '#e74c3c'],
        text=[f"{win_prob_a:.1%}", f"{win_prob_b:.1%}"],
        textposition='auto'
    ))
    fig.update_layout(
        title="Win Probability",
        xaxis_title="Probability",
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Radar-like comparison of key stats
    categories = ['Win %', 'Net Rating', 'Off Rating', 'Def Rating', 'Pace']
    
    # Normalize for radar
    def normalize(val, col):
        mn, mx = teams_df[col].min(), teams_df[col].max()
        return (val - mn) / (mx - mn) if mx != mn else 0.5
    
    vals_a = [
        normalize(stats_a["Win_Pct"], "Win_Pct"),
        normalize(stats_a["Net_Rating"], "Net_Rating"),
        normalize(stats_a["Off_Rating"], "Off_Rating"),
        1 - normalize(stats_a["Def_Rating"], "Def_Rating"),  # lower def rating is better
        normalize(stats_a["Pace"], "Pace")
    ]
    vals_b = [
        normalize(stats_b["Win_Pct"], "Win_Pct"),
        normalize(stats_b["Net_Rating"], "Net_Rating"),
        normalize(stats_b["Off_Rating"], "Off_Rating"),
        1 - normalize(stats_b["Def_Rating"], "Def_Rating"),
        normalize(stats_b["Pace"], "Pace")
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]], theta=categories + [categories[0]],
                                   fill='toself', name=team_a.split()[-1]))
    fig.add_trace(go.Scatterpolar(r=vals_b + [vals_b[0]], theta=categories + [categories[0]],
                                   fill='toself', name=team_b.split()[-1]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=True, height=450, title="Normalized Season Strength Profile")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""
    **Conceptual Dynamical Systems View** (from the original notes)
    
    In the full HyperCube-Nash engine:
    - Team states live on a high-dimensional hypercube (~65k+ configurations)
    - Krasnosel'skii-Mann iteration converges to fixed-point equilibria
    - Basins of attractors pull toward stable outcomes
    - Arnold tongues identify synchronization regions of team rhythms
    - Bayesian Metropolis-Hastings samples the posterior over outcomes
    """)
    
    # Simple attractor visualization
    fig = go.Figure()
    # Stronger team pulls toward higher win prob
    x = np.linspace(0, 1, 100)
    if stats_a["Net_Rating"] > stats_b["Net_Rating"]:
        attractor = win_prob_a
        label = f"{team_a.split()[-1]} attractor"
    else:
        attractor = win_prob_b
        label = f"{team_b.split()[-1]} attractor"
    
    # Simulated basin
    basin = np.exp(-((x - attractor)**2) / 0.02)
    fig.add_trace(go.Scatter(x=x, y=basin, fill='tozeroy', name="Basin of Attraction",
                             line=dict(color='#1f77b4')))
    fig.add_vline(x=attractor, line_dash="dash", line_color="red", 
                  annotation_text=label)
    fig.update_layout(title="Simplified Attractor Basin (Conceptual)",
                      xaxis_title="Win Probability Space",
                      yaxis_title="Attraction Strength",
                      height=350)
    st.plotly_chart(fig, use_container_width=True)

# Detailed stats table
with st.expander("Full Season Stats Comparison"):
    compare = pd.DataFrame({
        "Metric": ["Record", "Win %", "PPG", "OPP PPG", "Point Diff", "Net Rating", 
                   "Off Rating", "Def Rating", "Pace", "Elo"],
        team_a: [
            f"{int(stats_a['Wins'])}-{int(stats_a['Losses'])}",
            f"{stats_a['Win_Pct']:.1%}",
            f"{stats_a['PPG']:.1f}",
            f"{stats_a['OPP_PPG']:.1f}",
            f"{stats_a['Point_Diff']:+.1f}",
            f"{stats_a['Net_Rating']:+.1f}",
            f"{stats_a['Off_Rating']:.1f}",
            f"{stats_a['Def_Rating']:.1f}",
            f"{stats_a['Pace']:.1f}",
            f"{int(stats_a['Elo'])}"
        ],
        team_b: [
            f"{int(stats_b['Wins'])}-{int(stats_b['Losses'])}",
            f"{stats_b['Win_Pct']:.1%}",
            f"{stats_b['PPG']:.1f}",
            f"{stats_b['OPP_PPG']:.1f}",
            f"{stats_b['Point_Diff']:+.1f}",
            f"{stats_b['Net_Rating']:+.1f}",
            f"{stats_b['Off_Rating']:.1f}",
            f"{stats_b['Def_Rating']:.1f}",
            f"{stats_b['Pace']:.1f}",
            f"{int(stats_b['Elo'])}"
        ]
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("""
**Demo Notes:** This is a working prototype of the sports prediction app.  
It uses realistic mock season data and practical statistical methods (Elo + net rating).  
The "HyperCube-Nash" mode applies a conceptual attractor adjustment inspired by the original mathematical notes  
(basins of attractors, Krasnosel'skii-Mann fixed points, hypercube state space, etc.).  

In a production version you would connect live data APIs, implement the full numerical codex / tensor maps,  
and run proper MCMC sampling. This shows the core user experience: pick two teams → get win probability based on season stats.
""")
