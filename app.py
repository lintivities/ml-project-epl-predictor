import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Premier League Winner Predictor", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg, #f5f7fb 0%, #e9eefb 100%);
    color: #0f172a;
}
[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] *{
    color: white !important;
}
.block-container{
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.title{
    font-size: 2.6rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.subtitle{
    font-size: 1rem;
    color: #475569;
    margin-bottom: 1.4rem;
}
.panel{
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 22px;
    padding: 1.2rem 1.2rem;
    box-shadow: 0 10px 30px rgba(15,23,42,0.08);
}
.metric-card{
    background: white;
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 22px rgba(15,23,42,0.08);
    border: 1px solid rgba(148,163,184,0.14);
    min-height: 130px;
}
.metric-label{
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 0.45rem;
}
.metric-value{
    font-size: 1.95rem;
    font-weight: 800;
    color: #0f172a;
}
.hero-card{
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
    color: white;
    border-radius: 22px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 12px 28px rgba(37,99,235,0.28);
    min-height: 130px;
}
.hero-title{
    font-size: 0.95rem;
    opacity: 0.9;
}
.hero-value{
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.2rem;
}
.hero-sub{
    font-size: 1rem;
    margin-top: 0.35rem;
    opacity: 0.95;
}
.section-title{
    font-size: 1.2rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}
.contender-card{
    background: white;
    border-radius: 20px;
    padding: 1rem 1rem;
    box-shadow: 0 8px 22px rgba(15,23,42,0.08);
    border: 1px solid rgba(148,163,184,0.14);
    text-align: center;
    min-height: 145px;
}
.contender-rank{
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}
.contender-team{
    font-size: 1.1rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.25;
    margin-bottom: 0.45rem;
}
.contender-prob{
    font-size: 1.45rem;
    font-weight: 800;
    color: #2563eb;
}
.small-note{
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}
div[data-testid="stDataFrame"]{
    background: white;
    border-radius: 18px;
    padding: 0.4rem;
    box-shadow: 0 8px 22px rgba(15,23,42,0.06);
}
</style>
""", unsafe_allow_html=True)

team_season = pd.read_csv("team_season_features.csv")
model = joblib.load("gb_model.pkl")

feature_cols = [
    "matches",
    "points",
    "wins",
    "draws",
    "losses",
    "gf",
    "ga",
    "gd",
    "ppg",
    "win_ratio",
    "draw_ratio",
    "loss_ratio",
    "gd_per_match",
    "gf_per_match",
    "ga_per_match",
    "home_points",
    "away_points",
    "home_gd",
    "away_gd",
    "home_win_ratio",
    "away_win_ratio"
]

latest = team_season.sort_values("season").groupby("team").tail(1).copy()
latest = latest.sort_values("season").reset_index(drop=True)

teams = sorted(latest["team"].unique())

st.sidebar.markdown("## ⚙️ Controls")
selected_team = st.sidebar.selectbox("Select Team", teams)
target_year = st.sidebar.number_input("Select Target Year", min_value=2000, max_value=2035, value=2026, step=1)
mode = st.sidebar.radio("Prediction Mode", ["Use saved team profile", "Adjust team stats manually"])

base_row = latest[latest["team"] == selected_team].iloc[0]

matches = int(base_row["matches"])
points = int(base_row["points"])
wins = int(base_row["wins"])
draws = int(base_row["draws"])
losses = int(base_row["losses"])
gf = int(base_row["gf"])
ga = int(base_row["ga"])
home_points = int(base_row["home_points"])
away_points = int(base_row["away_points"])
home_gd = int(base_row["home_gd"])
away_gd = int(base_row["away_gd"])
home_win_ratio = float(base_row["home_win_ratio"])
away_win_ratio = float(base_row["away_win_ratio"])

if mode == "Adjust team stats manually":
    matches = st.sidebar.slider("Matches", 1, 38, matches)
    points = st.sidebar.slider("Points", 0, 114, points)
    wins = st.sidebar.slider("Wins", 0, 38, min(wins, 38))
    draws = st.sidebar.slider("Draws", 0, 38, min(draws, 38))
    losses = st.sidebar.slider("Losses", 0, 38, min(losses, 38))
    gf = st.sidebar.slider("Goals For", 0, 120, gf)
    ga = st.sidebar.slider("Goals Against", 0, 120, ga)
    home_points = st.sidebar.slider("Home Points", 0, 57, min(home_points, 57))
    away_points = st.sidebar.slider("Away Points", 0, 57, min(away_points, 57))
    home_gd = st.sidebar.slider("Home Goal Difference", -50, 80, home_gd)
    away_gd = st.sidebar.slider("Away Goal Difference", -50, 80, away_gd)
    home_win_ratio = st.sidebar.slider("Home Win Ratio", 0.0, 1.0, float(home_win_ratio), 0.01)
    away_win_ratio = st.sidebar.slider("Away Win Ratio", 0.0, 1.0, float(away_win_ratio), 0.01)

gd = gf - ga
ppg = points / matches if matches else 0
win_ratio = wins / matches if matches else 0
draw_ratio = draws / matches if matches else 0
loss_ratio = losses / matches if matches else 0
gd_per_match = gd / matches if matches else 0
gf_per_match = gf / matches if matches else 0
ga_per_match = ga / matches if matches else 0

input_row = pd.DataFrame([{
    "matches": matches,
    "points": points,
    "wins": wins,
    "draws": draws,
    "losses": losses,
    "gf": gf,
    "ga": ga,
    "gd": gd,
    "ppg": ppg,
    "win_ratio": win_ratio,
    "draw_ratio": draw_ratio,
    "loss_ratio": loss_ratio,
    "gd_per_match": gd_per_match,
    "gf_per_match": gf_per_match,
    "ga_per_match": ga_per_match,
    "home_points": home_points,
    "away_points": away_points,
    "home_gd": home_gd,
    "away_gd": away_gd,
    "home_win_ratio": home_win_ratio,
    "away_win_ratio": away_win_ratio
}])

selected_probability = float(model.predict_proba(input_row[feature_cols])[:, 1][0] * 100)

latest_for_ranking = latest.copy()
latest_for_ranking["probability"] = model.predict_proba(latest_for_ranking[feature_cols])[:, 1] * 100
latest_for_ranking["probability"] = latest_for_ranking["probability"].round(2)

selected_index = latest_for_ranking[latest_for_ranking["team"] == selected_team].index
if len(selected_index) > 0:
    latest_for_ranking.loc[selected_index[0], "probability"] = round(selected_probability, 2)

latest_for_ranking = latest_for_ranking.sort_values("probability", ascending=False).reset_index(drop=True)

predicted_winner = latest_for_ranking.iloc[0]["team"]
predicted_winner_prob = float(latest_for_ranking.iloc[0]["probability"])
selected_rank = int(latest_for_ranking.index[latest_for_ranking["team"] == selected_team][0] + 1)

st.markdown('<div class="title">Premier League Winner Predictor</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Choose a team, choose any year you want, and either use the saved profile or manually shape the season stats for a custom title prediction.</div>',
    unsafe_allow_html=True
)

top_a, top_b, top_c = st.columns([1, 1, 1.25])

with top_a:
    st.markdown(
        f'''
        <div class="metric-card">
            <div class="metric-label">{selected_team} Title Probability</div>
            <div class="metric-value">{selected_probability:.2f}%</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

with top_b:
    st.markdown(
        f'''
        <div class="metric-card">
            <div class="metric-label">Predicted Rank for {target_year}</div>
            <div class="metric-value">#{selected_rank}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

with top_c:
    st.markdown(
        f'''
        <div class="hero-card">
            <div class="hero-title">Top Predicted Winner</div>
            <div class="hero-value">{predicted_winner}</div>
            <div class="hero-sub">{predicted_winner_prob:.2f}% estimated title probability</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

if selected_team == predicted_winner:
    st.success(f"{selected_team} is currently the top model pick to win the Premier League in {target_year}.")
else:
    st.warning(f"{selected_team} is not the top pick right now. The model currently ranks {predicted_winner} above it for {target_year}.")

st.markdown('<div class="small-note">The year selector is now open to older years too. Still, the model only predicts from the stats you give it. A future year is just the prediction label unless you provide realistic team inputs for that year.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Top 4 Contenders</div>', unsafe_allow_html=True)

top4 = latest_for_ranking.head(4).reset_index(drop=True)
c1, c2, c3, c4 = st.columns(4)

for i, col in enumerate([c1, c2, c3, c4]):
    with col:
        st.markdown(
            f'''
            <div class="contender-card">
                <div class="contender-rank">Rank #{i+1}</div>
                <div class="contender-team">{top4.loc[i, "team"]}</div>
                <div class="contender-prob">{top4.loc[i, "probability"]:.2f}%</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

tab1, tab2, tab3, tab4 = st.tabs(["Prediction Input", "Leaderboard", "Charts", "Feature Importance"])

with tab1:
    st.markdown('<div class="section-title">Model Input Used</div>', unsafe_allow_html=True)
    display_input = input_row.copy()
    display_input.insert(0, "team", selected_team)
    display_input.insert(1, "target_year", target_year)
    st.dataframe(display_input.round(3), use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Full Leaderboard</div>', unsafe_allow_html=True)
    leaderboard = latest_for_ranking[["team", "probability", "points", "gd", "season"]].copy()
    leaderboard.columns = ["Team", "Win Probability (%)", "Points", "Goal Difference", "Latest Season Used"]
    st.dataframe(leaderboard, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Top 10 Probability Chart</div>', unsafe_allow_html=True)
    top10 = latest_for_ranking.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10["team"], top10["probability"])
    ax.set_xlabel("Champion Probability (%)")
    ax.set_ylabel("Team")
    ax.set_title(f"Top 10 Teams Most Likely to Win in {target_year}")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.markdown('<div class="section-title">Selected Team vs Top Pick</div>', unsafe_allow_html=True)
    compare_df = pd.DataFrame({
        "Team": [selected_team, predicted_winner],
        "Probability": [selected_probability, predicted_winner_prob]
    })
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(compare_df["Team"], compare_df["Probability"])
    ax2.set_ylabel("Champion Probability (%)")
    ax2.set_title("Probability Comparison")
    st.pyplot(fig2)

with tab4:
    st.markdown('<div class="section-title">Gradient Boosting Feature Importance</div>', unsafe_allow_html=True)
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    importance.plot(kind="barh", ax=ax3)
    ax3.set_xlabel("Importance")
    ax3.set_title("Which Features Influence the Model Most")
    st.pyplot(fig3)