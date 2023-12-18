import models
import joblib
import streamlit as st
import pandas as pd
import eventstox
import matplotlib.pyplot as plt
import plotly.express as px

conn = st.experimental_connection('ftdataconnection', type='sql')


def get_query_data(key: str):

    seasons = conn.query(
        """
    SELECT DISTINCT season FROM matches;
    """
    )

    col1, col2, = st.columns(2)
    with col1:
        selected_season = st.selectbox(
            label="Season",
            options=seasons,
            key=f"{key}_select_season",
        )

    matches = conn.query(
        f"""
    SELECT home_team, away_team, home_score, away_score, match_id
    FROM matches
    WHERE season = '{selected_season}'
    """
    )
    matches['match'] = matches.apply(
        lambda x: f"{x['home_team']} {x['home_score']} : {x['away_score']} {x['away_team']}",
        axis=1
    )
    season_dict = {
        '2018/2019': '1819',
        '2019/2020': '1920',
        '2020/2021': '2021',
    }
    with col2:
        selected_match = st.selectbox(
            label="Match",
            options=matches['match'],
            key=f"{key}_select_match"
        )
    selected_match_id = matches.loc[matches['match']
                                    == selected_match, 'match_id'].iloc[0]

    # get corresponding data
    selected_df = pd.read_csv(f"df_{season_dict[selected_season]}.csv")
    selected_df = selected_df.loc[selected_df['match_id'] == int(
        selected_match_id)]

    return selected_df


def get_shap(df):

    X, y = eventstox.df_to_X_y(df)
    X = models.process_X(X)

    shap_df = models.get_shap_by_feature(df)
    shap_actions_df = models.get_shap_by_action(df)

    return shap_df, shap_actions_df


def get_selected_series(shap_actions_df):
    selected_offensive_team = st.radio(
        label="Offensive Team",
        options=shap_actions_df['offensive_team'].unique(),
        horizontal=True
    )

    selected_series_id = st.slider(
        "Action Series",
        min_value=0,
        max_value=len(
            shap_actions_df.loc[
                shap_actions_df['offensive_team'] == selected_offensive_team
            ]
        ) - 1
    )

    selected_series = shap_actions_df.loc[
        shap_actions_df['offensive_team'] == selected_offensive_team
    ].iloc[selected_series_id]

    return selected_series


st.title("Identifying Goal Scoring Opportunities in WSL Games")

# load data
df_1819 = pd.read_csv("df_1819.csv")
df_1920 = pd.read_csv("df_1920.csv")
df_2021 = pd.read_csv("df_2021.csv")


# load models
lgb = joblib.load('lgb.joblib')


# intro
with open("./lightgbm_intro.md", 'r', encoding='utf-8') as file:
    lightgbm_intro = file.read()
    file.close()

st.markdown(lightgbm_intro)


# model training
with open("./model_training.md", 'r', encoding='utf-8') as file:
    model_training = file.read()
    file.close()

st.markdown(model_training)


# SHAP values
with open("./shap_intro.md", 'r', encoding='utf-8') as file:
    shap_intro = file.read()
    file.close()

st.markdown(shap_intro)


# match analysis
st.markdown("## Match Analysis")

df = get_query_data('match_results')
shap_df, shap_actions_df = get_shap(df)

# action series
st.markdown("### Action Series Analysis")
selected_series = get_selected_series(shap_actions_df)

st.pyplot(models.plot_shap_on_pitch(selected_series))
st.pyplot(models.plot_shap_barh(selected_series))

# Player evaluation
st.markdown("### Player Evaluation")

shap_per_action_df = models.get_shap_per_action_df(shap_actions_df)

st.plotly_chart(
    models.plot_player_mean_shap_by_match(
        shap_per_action_df
    ),
    use_container_width=True,
)

selected_player = st.selectbox(
    label="Player",
    options=shap_per_action_df['player'].unique()
)
st.pyplot(
    models.player_heatmap(selected_player, shap_per_action_df),
    use_container_width=True
)

# team performance
st.markdown("### Team Evaluation")

selected_team = st.radio(
    label='Team',
    options=shap_per_action_df['team'].unique(),
    horizontal=True,
)

st.pyplot(
    models.team_heatmap(
        team=selected_team,
        shap_per_action_df=shap_per_action_df
    ),
    use_container_width=True
)


