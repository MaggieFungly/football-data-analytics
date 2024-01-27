## Goal Scoring Opportunity Prediction with LightGBM

The presented model uses LightGBM to predict the probability of a goal occurring following a sequence of actions executed by both offensive and defensive teams.

### Terminology

- **Action Series**: A sequence comprising a shot and the preceding 10 on-the-ball actions. Excludes game stoppage events such as referee ball drops and game starts/ends.
- **Offensive Team**: The team executing the shot in the action series.
- **Defensive Team**: The opposing team in the action series.

### Feature Set

Inspired by the [SPADL](https://dl.acm.org/doi/10.1145/3292500.3330758) for football interpretation, the LightGBM model employs similar features to illustrate an attacking series. Each action preceding the shot encompasses the following features:

- `location_x` (*float*): X-axis location when the action occurs.
- `location_y` (*float*): Y-axis location when the action occurs.
- `end_location_x` (*float*): X-axis location where the event concludes (if applicable, otherwise identical to the location).
- `end_location_y` (*float*): Y-axis location where the event concludes (if applicable, otherwise identical to the location).
- `type` (*categorical*): Categorical feature denoting the action type from statsbomb data.
- `outcome` (*int*):
    - 1 if the action is successful.
    - 0 if the action is not successful.
- `team` (*int*):
    - 1 if the action is executed by the offensive team.
    - 0 if the action is executed by the defensive team.

Additionally, the features of the final shot action in the attacking series encompass:

- `location_x` (*float*): X-axis location when the action occurs.
- `location_y` (*float*): Y-axis location when the action occurs.
- `shot_angle` (*float*): Shot angle calculated using `atan2` based on the start and end locations of the shot.

The features of an action series are determined at the moment the shot is taken, thus avoiding data leakage.

- **Labels** (*int*): 
    - 1 if the shot results in a goal
    - 0 if the shot does not result in a goal
