import plotly.graph_objects as go
from sklearn.manifold import Isomap
import models
import joblib
import streamlit as st
import pandas as pd
import eventstox
import matplotlib.pyplot as plt
import plotly.express as px

matches = pd.read_csv("./goal_scoring_both_teams/matches.csv")

def get_query_match_data(key: str):

    seasons = matches['season'].unique().tolist()
    # Streamlit interface for season and match selection
    col1, col2, = st.columns(2)
    with col1:
        selected_season = st.selectbox(
            label="Season", options=seasons, key=f"{key}_select_season")

    # Formatting match information for display
    matches['match'] = matches.apply(
        lambda x: f"{x['home_team']} {x['home_score']} : {x['away_score']} {x['away_team']}", axis=1)

    # Season dictionary for file mapping
    season_dict = {'2018/2019': '1819',
                   '2019/2020': '1920', '2020/2021': '2021'}
    with col2:
        selected_match = st.selectbox(
            label="Match", options=matches['match'], key=f"{key}_select_match")
    selected_match_id = matches.loc[matches['match']
                                    == selected_match, 'match_id'].iloc[0]

    # Retrieve and filter data for the selected match
    selected_df = pd.read_csv(
        f"./goal_scoring_both_teams/df_{season_dict[selected_season]}.csv")
    selected_df = selected_df.loc[selected_df['match_id'] == int(
        selected_match_id)]

    return selected_df


def get_shap(df):
    # Preprocessing data and calculating SHAP values
    X, y = eventstox.df_to_X_y(df)
    X = models.process_X(X)
    shap_df = models.get_shap_by_feature(df)
    shap_actions_df = models.get_shap_by_action(df)

    return shap_df, shap_actions_df


def get_selected_series(shap_actions_df):
    # Streamlit interface for selecting offensive team and action series
    selected_offensive_team = st.radio(
        label="Offensive Team", options=shap_actions_df['offensive_team'].unique(), horizontal=True)
    selected_series_id = st.slider("Action Series", min_value=0, max_value=len(
        shap_actions_df.loc[shap_actions_df['offensive_team'] == selected_offensive_team]) - 1)

    selected_series = shap_actions_df.loc[shap_actions_df['offensive_team']
                                          == selected_offensive_team].iloc[selected_series_id]

    return selected_series


# Application title
st.title("Identifying Goal Scoring Opportunities in WSL Games")

# Load datasets for different seasons
df_1819 = pd.read_csv("./goal_scoring_both_teams/df_1819.csv")
df_1920 = pd.read_csv("./goal_scoring_both_teams/df_1920.csv")
df_2021 = pd.read_csv("./goal_scoring_both_teams/df_2021.csv")

# Load the machine learning model
lgb = joblib.load('./goal_scoring_both_teams/lgb.joblib')

# Introduction to LightGBM
with open("./goal_scoring_both_teams/lightgbm_intro.md", 'r', encoding='utf-8') as file:
    lightgbm_intro = file.read()
    file.close()
st.markdown(lightgbm_intro)

# Model training process
with open("./goal_scoring_both_teams/model_training.md", 'r', encoding='utf-8') as file:
    model_training = file.read()
    file.close()
st.markdown(model_training)

# Introduction to SHAP values
with open("./goal_scoring_both_teams/shap_intro.md", 'r', encoding='utf-8') as file:
    shap_intro = file.read()
    file.close()
st.markdown(shap_intro)

# Match Analysis Section
st.markdown("## Match Analysis")
df = get_query_match_data('match_results')
shap_df, shap_actions_df = get_shap(df)

# Action Series Analysis
st.markdown("### Action Series Analysis")
selected_series = get_selected_series(shap_actions_df)
st.pyplot(models.plot_shap_on_pitch(selected_series))
st.pyplot(models.plot_shap_barh(selected_series))

# Player Evaluation Section
st.markdown("### Player Evaluation")
shap_per_action_df = models.get_shap_per_action_df(shap_actions_df)
st.plotly_chart(models.plot_player_mean_shap_by_match(
    shap_per_action_df), use_container_width=True)

selected_player = st.selectbox(
    label="Player", options=shap_per_action_df['player'].unique())
st.pyplot(models.player_heatmap(selected_player,
          shap_per_action_df), use_container_width=True)

# Individual Player Analysis
st.markdown("## Player Analysis")

with open("./goal_scoring_both_teams/clustering.md", 'r', encoding='utf-8') as file:
    clustering_intro = file.read()
    file.close()
st.markdown(clustering_intro)


selected_season = st.selectbox(
    "Season", ['2018/2019', '2019/2020', '2020/2021'])
season_dict = {
    '2018/2019': '1819',
    '2019/2020': '1920',
    '2020/2021': '2021'
}
df = pd.read_csv(
    f"./goal_scoring_both_teams/df_{season_dict[selected_season]}.csv")

st.markdown("### Clustering")
fig = models.player_clustering_plot(df)
st.plotly_chart(fig, use_container_width=True)

st.markdown('### Player SHAP Heatmap')
shap_actions_df = models.get_shap_by_action(df)
shap_per_action_df = models.get_shap_per_action_df(shap_actions_df)
selected_team = st.selectbox("Team", shap_per_action_df['team'].unique())
selected_player = st.selectbox(
    "Player", shap_per_action_df.loc[shap_per_action_df['team']==selected_team, 'player'].unique())
fig = models.player_shap_heatmap(selected_player, shap_per_action_df)
st.pyplot(fig)
