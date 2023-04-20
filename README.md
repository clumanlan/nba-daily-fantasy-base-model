# NBA Daily Fantasy

## Introduction
Creating a winning NBA fantasy lineup requires both: (accurate) fantasy point predictions and a way to create a lineup that maximizes the chances of winning. 
<br>
<br>
For fantasy point predictions, we create both a Random Forest and Linear Regression that predicts fantasy points based on rolling lagged player and team stats. We use these fantasy point predictions in the optimization and also require the constraint that at least two correlated players are included in the lineup in order to increase variance thereby increasing the upper range of total expected fantasy points for the lineup.
<br>
<br>


## Codebase
### Data Flow
![alt text](https://lucid.app/publicSegments/view/0422e716-7a97-424d-8479-4fc30e19a408/image.png)
<br>
<br>
The project consists of the following python scripts:
<br>
- **0-get-initial-data:** Pulls initial boxscore data (including advanced stats) starting from 2001 at the game, team and player level from [nba_api](https://github.com/swar/nba_api).
- **1-build-fp-pred-models:** Create and train Linear Regression and Random Forest models using lagged player and team features (e.g. Player Fantasy Point ranking, Player PACE, Opposing Team Fantasy Point Allowed).
- **2-get-rolling-data:** Pull same data as '0-get-initial-data' but on a rolling basis to get most recent game data and write to S3.
- **3-rolling-model-data-processing:** Process most recent data similar to '1-build-fp-pred-models', pull in today's game starters from Rotowire to get features needed before gametime (e.g. active players, Starter) and use data to predict player's fantasy point.
- **4-current-season-correlations:** Build correlations between players' fantasy point performance for the current season.
- **5-streamlit-dash:** Streamlit dash takes historical data, processed data and model output to create an optimized lineup that requires at least two correlated players to be included. Also, provides tools to view player's historic fantasy performance/trend and option to remove players from lineup.


## Streamlit Dash
 [NBA Daily Fantasy Model Dash](https://clumanlan-nba-daily-fantasy-base-model-5-streamlit-dash-p293dc.streamlit.app/)
<br>
![alt text](images/nba-base-model-streamlit-dash.png)



