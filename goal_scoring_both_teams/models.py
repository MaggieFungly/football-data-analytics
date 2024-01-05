import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import joblib
import shap
import eventstox
import matplotlib
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter


class Colors:

    yale_blue = "#033270"
    green_blue = "#1368aa"
    celestial_blue = "#4091c9"
    light_blue = "#9dcee2"
    misty_rose = "#fedfd4"
    salmon = "#f29479"
    tomato = "#f26a4f"
    vermilion = "#ef3c2d"
    engineering_orange = "#cb1b16"
    rosewood = "#65010c"


def process_X(X):

    # transform categorical features
    categorical_features = X.select_dtypes(
        exclude=['number']
    ).columns.tolist()

    X[categorical_features] = X[categorical_features].astype('category')

    return X


def get_shap_by_feature(df: pd.DataFrame):

    d = df.copy()

    X, y = eventstox.df_to_X_y(d)

    X = process_X(X)

    d['offensive_team'] = d['team_10'].values
    d['defensive_team'] = d[
        [f'team_{i}' for i in range(11)]
    ].apply(
        lambda x:
        np.unique(x)[np.unique(x) != x['team_10']][0]
        if len(np.unique(x)) == 2
        else 0,
        axis=1
    )

    # shap
    lgb = joblib.load('./goal_scoring_both_teams/lgb.joblib')
    explainer = shap.TreeExplainer(lgb)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values[1], columns=X.columns)

    columns_to_add = ['match_id',
                      'offensive_team',
                      'defensive_team',
                      'shot_id'
                      ]
    shap_df[columns_to_add] = d[columns_to_add]
    shap_df['probability'] = lgb.predict(X)

    return shap_df


def sum_shap_by_action(shap_df: pd.DataFrame):
    # for action 0 - 10, sum the corresponding items

    shap_actions_df = pd.DataFrame()

    for i in range(10):
        shap_cols = [f'team_{i}',
                     f'location_x_{i}',
                     f'location_y_{i}',
                     f'end_location_x_{i}',
                     f'end_location_y_{i}',
                     f'type_{i}',
                     f'outcome_{i}'
                     ]
        shap_actions_df[f'shap_{i}'] = shap_df[shap_cols].sum(
            axis=1, numeric_only=True)

    # the last shot action
    shap_actions_df['shap_10'] = shap_df[
        ['team_10',
         'location_x_10',
         'location_y_10',
         'shot_angle']
    ].sum(axis=1, numeric_only=True)

    # probability
    shap_actions_df['probability'] = shap_df['probability']

    return shap_actions_df


def get_team_status(x: pd.Series):

    teams = x[[f'team_{i}' for i in range(11)]]
    team_lst = list(set(teams))

    offensive_team = x['team_10']

    if len(team_lst) > 1:
        defensive_team = [i for i in team_lst if i != offensive_team][0]
    else:
        defensive_team = None

    x['offensive_team'] = offensive_team
    x['defensive_team'] = defensive_team

    return x


def get_shap_by_action(df: pd.DataFrame):

    df = df.reset_index(drop=True)

    shap_df = get_shap_by_feature(df)
    shap_actions_df = sum_shap_by_action(shap_df=shap_df)

    shap_actions_df = pd.concat(
        [shap_actions_df, df],
        axis=1,
    )

    shap_actions_df = shap_actions_df.apply(
        lambda x: get_team_status(x),
        axis=1
    )

    return shap_actions_df


def is_goal(outcome):
    if outcome == 0:
        return 'No Goal'
    else:
        return 'Goal'


def plot_shap_barh(actions):
    """
    Example: 

    ```python
    plot_shap_barh(shap_actions_df.iloc[0])
    ```

    Plot an action series' performance evaluated by SHAP values in horizontal barplots.

    """

    x = actions.copy()

    colors = [Colors.celestial_blue if team == x['offensive_team']
              else Colors.tomato for team in x[[f'team_{i}' for i in range(11)]]]

    fig, ax = plt.subplots(figsize=(8, 6))

    # shap values
    ax.barh(width=x[[f'shap_{i}' for i in range(11)]],
            y=np.arange(11),
            color=colors,
            )

    # players
    ax.set_yticks(np.arange(11))
    ax.set_yticklabels(
        labels=x[[f'player_{i}' for i in range(11)]],
    )

    # align the axes
    ax1 = ax.twinx()

    l = ax.get_ylim()
    l2 = ax1.get_ylim()

    def f(x): return l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
    ticks = f(ax.get_yticks())
    ax1.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax1.set_yticklabels(
        labels=x[[f'type_{i}' for i in range(11)]].values,
        verticalalignment='center',
    )

    # set yaxis to top-to-bottom
    ax.invert_yaxis()
    ax1.invert_yaxis()

    # title
    ax.set_title(is_goal(x['outcome_10']))
    ax.set_title(
        f"xG: {round(x['shot_statsbomb_xg'], 2)}\nProbability: {round(x['probability'], 2)}",
        loc='left',
        color='dimgray',
        fontweight='bold',
    )

    # set legends
    celestial_blue_patch = mpatches.Patch(
        color=Colors.celestial_blue,
        label=f"Offensive: {x['offensive_team']}"
    )
    tomato_patch = mpatches.Patch(
        color=Colors.tomato,
        label=f"Defensive: {x['defensive_team']}"
    )

    fig.legend(handles=[celestial_blue_patch, tomato_patch],
               loc='upper right',
               )

    return fig


def plot_shap_on_pitch(actions):
    """

    Plot corresponding action series on the pitch.

    """

    x = actions.copy()
    pitch = Pitch(pitch_color='white', line_color='gray')

    fig, ax = pitch.draw(figsize=(10, 8))

    team_colors = [Colors.celestial_blue if x[f'team_{i}'] ==
                   x[f'team_{10}'] else Colors.tomato for i in range(11)]

    for i in range(11):
        pitch.arrows(
            x[f'location_x_{i}'],
            x[f'location_y_{i}'],
            x[f'end_location_x_{i}'],
            x[f'end_location_y_{i}'],
            color=team_colors[i],
            ax=ax,
            width=2
        )
        pitch.scatter(
            x[f'location_x_{i}'],
            x[f'location_y_{i}'],
            ax=ax,
            s=10,
            color=team_colors[i]
        )
        pitch.text(
            x[f'location_x_{i}'] + 1,
            x[f'location_y_{i}'],
            s=str(i),
            color='dimgray',
            fontweight='bold',
            ax=ax,
        )

    ax.set_title(is_goal(x['outcome_10']))
    ax.set_title(
        f"xG: {round(x['shot_statsbomb_xg'], 2)}\nProbability: {round(x['probability'], 2)}",
        loc='left',
        color='dimgray',
        fontweight='bold',
    )

    # set legends
    offensive_patch = mpatches.Patch(
        color=Colors.celestial_blue,
        label=f"Offensive: {x['offensive_team']}"
    )
    defensive_patch = mpatches.Patch(
        color=Colors.tomato,
        label=f"Defensive: {x['defensive_team']}"
    )

    fig.legend(handles=[offensive_patch, defensive_patch],
               loc='upper right',
               )

    return fig


def melt_shap_actions(row):
    # Create a DataFrame from the row
    d = pd.DataFrame({
        'player': row[[f'player_{i}' for i in range(11)]].values,
        'team': row[[f'team_{i}' for i in range(11)]].values,
        'shap': row[[f'shap_{i}' for i in range(11)]].values,
        'type': row[[f'type_{i}' for i in range(11)]].values,
        'location_x': row[[f'location_x_{i}' for i in range(11)]].values,
        'location_y': row[[f'location_y_{i}' for i in range(11)]].values,
        'end_location_x': row[[f'end_location_x_{i}' for i in range(11)]].values,
        'end_location_y': row[[f'end_location_y_{i}' for i in range(11)]].values,
        'offensive_team': row['offensive_team'],
        'defensive_team': row['defensive_team'],
        'shot_id': row['shot_id'],
        'match_id': row['match_id']
    })
    return d


def get_shap_per_action_df(shap_actions_df):
    # Use apply to transform each row and concatenate the results
    shap_per_action_df = pd.concat(shap_actions_df.apply(
        melt_shap_actions, axis=1).values, ignore_index=True)
    return shap_per_action_df


def get_player_shap(player: str, shap_per_action_df: pd.DataFrame):
    """
    Get all the actions performed by a player
    """

    # get actions of the player from shap_per_action_df
    player_df = shap_per_action_df.loc[
        shap_per_action_df['player'] == player
    ].copy()

    # transform shap if defensive
    player_df['shap'] = player_df.apply(
        lambda x:
        -x['shap'] if x['team'] == x['defensive_team']
        else x['shap'],
        axis=1
    ).values

    # transform location if defensive
    player_df[['location_x', 'end_location_x']] = player_df.apply(
        lambda x: 120 - x[['location_x', 'end_location_x']]
        if x['team'] == x['defensive_team']
        else x[['location_x', 'end_location_x']],
        axis=1
    )

    player_df[['location_y', 'end_location_y']] = player_df.apply(
        lambda x: 80 - x[['location_y', 'end_location_y']]
        if x['team'] == x['defensive_team']
        else x[['location_y', 'end_location_y']],
        axis=1
    )

    player_df = player_df.reset_index(drop=True)

    return player_df


def player_summary(player_df: pd.DataFrame):

    # get player summary by type
    player_df = player_df.copy()
    player_df['is_offensive'] = player_df.apply(
        lambda x: 0 if x['team'] == x['defensive_team']
        else 1,
        axis=1
    )
    player_df_grouped = player_df.groupby(
        ['type', 'is_offensive']
    )['shap'].agg(['mean', 'std', 'count'])

    player_df_grouped = player_df_grouped.reset_index()
    player_df_grouped = player_df_grouped.sort_values(
        by='count', ascending=False).reset_index(drop=True)

    # plot summary
    fig, ax = plt.subplots(figsize=(8, 7))

    for is_offensive in [0, 1]:

        performance = player_df_grouped.loc[
            (player_df_grouped['is_offensive'] == is_offensive)
        ]
        if is_offensive == 0:
            abs_mean = -abs(performance['mean'])
        else:
            abs_mean = abs(performance['mean'])

        # barplot
        ax.barh(
            y=performance['type'],
            width=abs_mean,
            color=[Colors.celestial_blue if mean >= 0
                   else Colors.tomato
                   for mean in performance['mean']
                   ]
        )

    ax.invert_yaxis()

    ax.set_title('Defensive', loc='left')
    ax.set_title('Offensive', loc='right')

    fig.suptitle(
        f"{player_df['player'].iloc[0]} - {player_df['team'].iloc[0]}"
    )

    positive_patch = mpatches.Patch(
        color=Colors.celestial_blue,
        label=f"Positive"
    )
    negative_patch = mpatches.Patch(
        color=Colors.tomato,
        label=f"Negative"
    )

    fig.legend(handles=[positive_patch, negative_patch],
               loc='upper right'
               )

    return player_df_grouped, fig


def aggregate(d: pd.DataFrame, num_actions: int = 3):

    player = d['player'].iloc[0]

    # Vectorized calculation of 'is_offensive' column
    d['is_offensive'] = np.where(d['team'] == d['offensive_team'], 1, 0)

    # Select only necessary columns
    d = d[['shap', 'type', 'location_x', 'location_y',
           'end_location_x', 'end_location_y', 'is_offensive']]

    # Groupby aggregation using a dictionary
    agg_dict = {
        'shap': ['mean', 'std'],
        'location_x': ['mean', 'std'],
        'location_y': ['mean', 'std'],
        'end_location_x': ['mean', 'std'],
        'end_location_y': ['mean', 'std'],
        'type': 'count',
    }
    d_grouped = d.groupby(['type', 'is_offensive']).agg(agg_dict).reset_index()

    # Get the top num_actions rows without sorting the entire DataFrame
    d_grouped = d_grouped.nlargest(num_actions, ('type', 'count')).melt()

    # Use f-strings for string concatenation
    d_grouped['index'] = d_grouped.apply(
        lambda x: f'{x["variable_0"]}_{x["variable_1"]}' if x['variable_1'] != '' else x['variable_0'], axis=1)

    counts = {}
    result = []
    for item in d_grouped['index']:
        counts[item] = counts.get(item, 0) + 1
        result.append(f'{item}_{counts[item]}')

    d_grouped['index'] = result

    s = pd.DataFrame(
        dict(zip(d_grouped['index'], d_grouped['value'])), index=[player])
    return s


def get_player_top_actions(shap_per_action_df: pd.DataFrame):
    """

    Get each player's repersentative types of actions. Information inlcudes:
    - locations: mean, std
    - type of actions: count
    - offensive or defensive for each type of action

    """

    player_top_actions_df = shap_per_action_df.groupby(
        'player').apply(aggregate)

    players = player_top_actions_df.index.get_level_values(0).values
    player_top_actions_df = player_top_actions_df.set_index(players)

    # drop na's
    player_top_actions_df = player_top_actions_df.dropna()

    return player_top_actions_df


def encode_standardize(X: pd.DataFrame, encoder=None, scaler=None):

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(X[categorical_features])
    if scaler is None:
        scaler = MinMaxScaler((0, 1))
        scaler.fit(X[numerical_features])

    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()

    X_scaled = pd.DataFrame()

    X_scaled[numerical_features] = scaler.transform(X[numerical_features])
    X_scaled[encoder.get_feature_names_out(
    )] = encoder.transform(X[categorical_features])

    return X_scaled, encoder, scaler


def plot_player_mean_shap_by_match(shap_per_action_df):
    """Returns a scatter plot of each player's mean SHAP value and number of actions in a match."""

    shap_grouped_by_player_df = shap_per_action_df.groupby(
        ['player', 'team'])['shap'].agg(
        ['mean', 'count']
    ).reset_index()

    reference_line_mean = (
        shap_grouped_by_player_df['count'] * shap_grouped_by_player_df['mean']).mean()

    fig = go.Figure(
        data=[
            go.Scatter(
                x=shap_grouped_by_player_df['count'],
                y=shap_grouped_by_player_df['mean'],
                mode='markers',
                text=[
                    f"Player: {player}<br>Team: {team}"
                    for player, team in zip(
                        shap_grouped_by_player_df['player'],
                        shap_grouped_by_player_df['team'],
                    )
                ],
                hovertemplate='%{text}<br>Number of actions: %{x}<br>Mean shap: %{y:.2f}',
                marker=dict(
                    color=[
                        Colors.tomato if
                        team == shap_grouped_by_player_df['team'].iloc[0]
                        else Colors.celestial_blue
                        for team in shap_grouped_by_player_df['team']
                    ],
                    size=shap_grouped_by_player_df['count'],
                ),
                name="",
            ),
        ]
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


def player_heatmap(player: str, shap_per_action_df: pd.DataFrame):
    "Plot the player's heatmap on the pitch."

    player_shap_df = get_player_shap(
        player=player,
        shap_per_action_df=shap_per_action_df
    )

    pitch = Pitch(
        pitch_color='white',
        line_color='gray',
        line_zorder=2,
    )
    fig, ax = pitch.draw()

    bin_statistic = pitch.bin_statistic(
        player_shap_df['location_x'],
        player_shap_df['location_y'],
        statistic='count',
        bins=(30, 30),
    )

    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap='gist_heat_r',
        edgecolors=None
    )

    fig.suptitle(f"Player Heatmap - {player}")
    ax.set_title(
        f"Mean SHAP: {round(np.mean(player_shap_df['shap']), 2)}",
        loc='left',
        color='dimgray',
        fontweight='bold',
        fontsize=9,
    )

    return fig


def team_heatmap(team: str, shap_per_action_df: pd.DataFrame):
    """
    Plot the player's heatmap on the pitch.
    """

    team_shap_df = shap_per_action_df.loc[shap_per_action_df['team'] == team]
    team_shap_df[['location_x', 'location_y']] = team_shap_df[[
        'location_x', 'location_y']].astype('float')

    pitch = Pitch(
        pitch_color='white',
        line_color='gray',
        line_zorder=2,
    )
    fig, ax = pitch.draw()

    bin_statistic = pitch.bin_statistic(
        team_shap_df['location_x'],
        team_shap_df['location_y'],
        statistic='count',
        bins=(30, 30),
    )

    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)

    pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap='gist_heat_r',
        edgecolors=None
    )

    fig.suptitle(f"{team}")
    ax.set_title(
        f"Mean SHAP: {round(np.mean(team_shap_df['shap']), 2)}",
        loc='left',
        color='dimgray',
        fontweight='bold',
        fontsize=9,
    )

    return fig


def player_clustering_plot(df):

    shap_actions_df = get_shap_by_action(df)
    shap_per_action_df = get_shap_per_action_df(shap_actions_df)
    player_top_actions_df = get_player_top_actions(shap_per_action_df)

    encoder = joblib.load('./goal_scoring_both_teams/encoder.joblib')
    scaler = joblib.load('./goal_scoring_both_teams/scaler.joblib')
    isomap = joblib.load('./goal_scoring_both_teams/isomap.joblib')
    em = joblib.load('./goal_scoring_both_teams/em.joblib')

    X_scaled, encoder, scaler = encode_standardize(
        player_top_actions_df, encoder=encoder, scaler=scaler)
    X_visualized = isomap.transform(X_scaled)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=X_visualized[:, 0],
                y=X_visualized[:, 1],
                mode='markers',
                marker=dict(
                    color=em.predict(X_scaled),
                ),
                text=player_top_actions_df.index,
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig


def player_shap_heatmap(player_name, shap_per_action_df):
    
    player_df = get_player_shap(player_name, shap_per_action_df)

    pitch = Pitch(
        pitch_color='white',
        line_color='gray',
        line_zorder=2,
    )

    fig, ax = pitch.draw()
    bin_statistic = pitch.bin_statistic(
        x=player_df['location_x'],
        y=player_df['location_y'],
        values=player_df['shap'],
        bins=(30, 30),
    )
    heatmap = pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap='RdBu',
        edgecolors=None,
        vmin=player_df['shap'].min(),
        vmax=player_df['shap'].max(),
    )

    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('SHAP Value')

    ax.set_title(f"Player SHAP Heatmap - {player_name}")

    return fig