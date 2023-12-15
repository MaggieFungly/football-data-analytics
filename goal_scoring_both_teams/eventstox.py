from datetime import timedelta
import ast
import pandas as pd
import numpy as np
import math


def inquire_col(col, df):
    return [i for i in df.columns if col in i]


def type_handling(events):
    # Sort and reset index
    events = events.sort_values(
        by=['match_id', 'index']
    ).reset_index(drop=True)

    # Convert timestamp
    events[['minute', 'second']] = events[['minute', 'second']].astype(int)
    events['milliseconds'] = events['timestamp'].apply(
        lambda x: int(x.split('.')[1])
    )
    events['timestamp'] = events.apply(
        lambda x: timedelta(
            minutes=x['minute'], seconds=x['second'], milliseconds=x['milliseconds']
        ),
        axis=1)

    # Extract location
    events['location_x'] = events['location'].apply(
        lambda x: ast.literal_eval(x)[0] if x is not None else x)
    events['location_y'] = events['location'].apply(
        lambda x: ast.literal_eval(x)[1] if x is not None else x)

    # Extract end_location
    events['end_location'] = events[inquire_col('end_location', events)].apply(
        lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan, axis=1)
    events['end_location_x'] = events['end_location'].apply(
        lambda x: ast.literal_eval(x)[0] if pd.notna(x) else x)
    events['end_location_y'] = events['end_location'].apply(
        lambda x: ast.literal_eval(x)[1] if pd.notna(x) else x)

    # Fill missing end_location values
    events.loc[pd.isna(events['end_location_x']), 'end_location_x'] = events.loc[
        pd.isna(events['end_location_x']), 'location_x']
    events.loc[pd.isna(events['end_location_y']), 'end_location_y'] = events.loc[
        pd.isna(events['end_location_y']), 'location_y']

    # Convert shot_statsbomb_xg
    events['shot_statsbomb_xg'] = events['shot_statsbomb_xg'].apply(
        lambda x: ast.literal_eval(x) if x is not None else x)

    return events


def get_outcomes(events):

    # prepare columns
    events['outcome'] = np.nan

    events['next_action'] = events.groupby('shot_id').shift(-1)['type'].values
    events['next_team'] = events.groupby('shot_id').shift(-1)['team'].values
    events['next_possession_team'] = events.groupby(
        'shot_id').shift(-1)['possession_team'].values

    def get_action_outcome(advantage_condition, disadvantage_condition):
        events.loc[(advantage_condition), 'outcome'] = 1
        events.loc[(disadvantage_condition), 'outcome'] = 0

    # ball receipt
    get_action_outcome(
        (events['type'] == 'Ball Receipt*') & ~(events['ball_receipt_outcome']
                                                == 'Incomplete'),
        (events['type'] ==
         'Ball Receipt*') & (events['ball_receipt_outcome'] == 'Incomplete')
    )

    # dribble
    get_action_outcome(
        (events['type'] == 'Dribble') & (
            events['dribble_outcome'] == 'Complete'),
        (events['type'] == 'Dribble') & (events['dribble_outcome'] == 'Incomplete'))

    # duel
    # fail: duel_outcome fails or team loses possession
    get_action_outcome(
        (events['type'] == 'Duel')
        & (
            (events['duel_outcome'].isin(
                ['Success In Play', 'Won', 'Success Out']))
            | (events['team'] == events['next_possession_team'])
        ),
        (events['type'] == 'Duel')
        & (
            (events['duel_outcome'].isin(
                ['Lost In Play', 'Lost Out']))
            | (events['team'] != events['next_possession_team'])
        )
    )

    # goal keeper
    get_action_outcome(
        (events['type'] == 'Goal Keeper')
        & (events['goalkeeper_outcome'].isin(
            ['Clear', 'In Play Safe', 'Success', 'Claim', 'Saved Twice', 'Punched out', 'Won', None])),
        (events['type'] == 'Goal Keeper')
        & (events['goalkeeper_outcome'].isin(
            ['Fail', 'In Play Danger',
             'Touched Out', 'No Touch', 'Lost In Play', 'Touched In']))
    )

    # interception
    get_action_outcome(
        (events['type'] == 'Interception')
        & (events['interception_outcome'].isin(
            ['Won', 'Success In Play'])),
        (events['type'] == 'Interception')
        & (events['interception_outcome'].isin(
            ['Lost In Play', 'Lost Out']))
    )

    # pass
    get_action_outcome(
        (events['type'] == "Pass")
        & ~(events['pass_outcome'].isin(
            ['Incomplete', 'Pass Offside',
             'Out', 'Injury Clearance'])
            ),
        (events['type'] == "Pass")
        & (events['pass_outcome'].isin(
            ['Incomplete', 'Pass Offside', 'Out', 'Injury Clearance']))
    )

    # shot
    get_action_outcome(
        (events['type'] == 'Shot')
        & (events['shot_outcome'].isin(['Goal'])),
        (events['type'] == 'Shot')
        & ~(events['shot_outcome'].isin(['Goal']))
    )

    # substitution
    get_action_outcome(
        (events['type'] == 'Substitution')
        & (events['substitution_outcome'].isin(['Tactical'])),
        (events['type'] == 'Substitution')
        & (events['substitution_outcome'].isin(['Injury']))
    )

    # block
    get_action_outcome(
        (events['type'] == 'Block')
        & (
            (events['block_offensive'] == 'True')
            | (events['team'] == events['next_possession_team'])
        ),
        (events['type'] == 'Block')
        & (
            (events['block_deflection'] == 'True')
            | (events['team'] != events['next_possession_team'])
        )
    )

    # ball recovery
    get_action_outcome(
        (events['type'] == 'Ball Recovery')
        & ~(events['ball_recovery_recovery_failure'] == 'True'),
        (events['type'] == 'Ball Recovery')
        & (events['ball_recovery_recovery_failure'] == 'True')
    )

    # 50/50
    # success: team wins possession (team != possession_team & team = next_possession_team)
    # fail: team loses possession (team = possession_team & team != next_possession_team)
    get_action_outcome(
        (events['type'] == '50/50')
        & (
            (
                (events['possession_team'] == events['next_possession_team'])
                & (events['team'] == events['possession_team']))
            | (
                (events['possession_team'] != events['next_possession_team'])
                & (events['team'] != events['possession_team']))),
        (events['type'] == '50/50')
        & (
            (
                (events['possession_team'] == events['next_possession_team'])
                & (events['team'] != events['possession_team'])
            )
            | (
                (events['possession_team'] != events['next_possession_team'])
                & (events['team'] == events['possession_team'])
            )
        )
    )

    # negative actions
    negative_actions = ['Miscontrol',
                        'Foul Committed',
                        'Dispossessed',
                        'Dribbled Past',
                        'Bad Behaviour',
                        'Error',
                        'Injury Stoppage',
                        ]
    events.loc[events['type'].isin(negative_actions), 'outcome'] = 0

    # positive actions
    positive_actions = ['Clearance',
                        'Foul Won',
                        'Carry',
                        'Pressure',
                        'Shield',
                        ]
    events.loc[events['type'].isin(positive_actions), 'outcome'] = 1

    # neutral actions
    neutral_actions = ['Referee Ball-Drop',
                       'Player Off',
                       'Player On',
                       'Tactical Shift',
                       'Starting XI',
                       'Half Start',
                       ]
    events.loc[events['type'].isin(neutral_actions), 'outcome'] = -1

    return events


def get_action_series(actions, convert_location: bool):

    action_series = dict()
    for i in range(11):
        action_series[f'team_{i}'] = actions['team'].iloc[i]

        action_series[f'location_x_{i}'] = actions['location_x'].iloc[i]
        action_series[f'location_y_{i}'] = actions['location_y'].iloc[i]
        action_series[f'end_location_x_{i}'] = actions['end_location_x'].iloc[i]
        action_series[f'end_location_y_{i}'] = actions['end_location_y'].iloc[i]

        action_series[f'type_{i}'] = actions['type'].iloc[i]
        action_series[f'outcome_{i}'] = actions['outcome'].iloc[i]
        action_series[f'player_{i}'] = actions['player'].iloc[i]

        if convert_location == True:
            if action_series[f'team_{i}'] != actions['team'].iloc[10]:
                action_series[f'location_x_{i}'] = 120 - \
                    actions['location_x'].iloc[i]
                action_series[f'location_y_{i}'] = 80 - \
                    actions['location_y'].iloc[i]
                action_series[f'end_location_x_{i}'] = 120 - \
                    actions['end_location_x'].iloc[i]
                action_series[f'end_location_y_{i}'] = 80 - \
                    actions['end_location_y'].iloc[i]

    action_series['match_id'] = actions['match_id'].iloc[0]
    action_series['shot_id'] = actions['shot_id'].iloc[0]
    action_series['shot_statsbomb_xg'] = actions['shot_statsbomb_xg'].iloc[10]

    return pd.DataFrame(action_series, index=[0])


def events_to_df(events: pd.DataFrame, is_convert_location=True):
    events = type_handling(events)
    events = get_outcomes(events)

    df = events.groupby('shot_id').apply(
        lambda x: get_action_series(x, convert_location=is_convert_location)).reset_index(drop=True)

    return df


def df_to_X_y(df: pd.DataFrame):

    y = df['outcome_10'].values
    X = df.copy()

    # convert teams
    for i in range(11):
        X[f'team_{i}'] = X.apply(
            lambda x: 1 if x[f'team_{i}'] == x['team_10']
            else 0,
            axis=1
        )

    # shot angle
    X['shot_angle'] = X.apply(
        lambda x: math.atan2(
            x['end_location_y_10'] - x['location_y_10'],
            x['end_location_x_10'] - x['location_x_10']
        ),
        axis=1
    )

    # drop columns
    X = X.drop(
        columns=['match_id', 'shot_id']
        + [f'player_{i}' for i in range(11)]
        + ['end_location_x_10',
           'end_location_y_10',
            'outcome_10',
            'type_10',
            'shot_statsbomb_xg']
    )

    X = X.reset_index(drop=True)

    # dropna
    X = X.dropna()
    y = y[X.index.tolist()]
    X = X.reset_index(drop=True)

    return X, y
