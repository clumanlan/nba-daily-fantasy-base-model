import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import awswrangler as wr
from st_aggrid import AgGrid

st.set_page_config(page_title="Current Fantasy Lineup", layout="wide") 
st.title("Current Fantasy Lineup")

# Load and prep data -----------------------------------------------
today = date.today().strftime('%Y-%m-%d')

pred_path = f's3://nbadk-model/predictions/base/linear_regression/'

player_pred = wr.s3.read_parquet(
        path=pred_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

player_pred_rel_cols = ['PLAYER_NAME', 'POSITION', 'starter', 'fantasy_point_prediction',
                        'fantasy_points_rank_overall_lagged_team', 
                        'player_fantasy_points_rank_overall_lagged','fantasy_points_rank_overall_lagged_team_allowed_opposing']

cols = pd.DataFrame(player_pred.columns)

player_pred_filtered = player_pred[player_pred_rel_cols]
player_pred_filtered.columns =  ['Player Name', 'Position', 'Starter', 'FantasyPoint Prediction',
                        'Player Fantasy Points Rank Overall',
                        'Opposing Team Fantasy Points Allowed Rank Overall']


# PULL IN DRAFTKINGS ROSTER TO GET AVG POINTS AND SALARIES 
# PULL IN 

    

st.table(player_pred_filtered)



# PLOTS OF PLAYER VERSUS PREDICTED PERFORMANCE ------------------------------------


# Validate model ------------------------------
boxscore_validation = combined_player_team_boxscore[combined_player_team_boxscore['SEASON_ID']==2019]
boxscore_validation = boxscore_validation.dropna()

# Load Linear Regression model
lr_run_id = '29f299a29533425bb969722eceb5929d'
model_uri = f"projects/nba-daily-fantasy-base-model/mlruns/1/{lr_run_id}/artifacts/LinearRegression"
lr_base = mlflow.sklearn.load_model(model_uri)



# Apply prediction to boxscore validation set
boxscore_validation['fantasy_points_pred'] = lr_base.predict(boxscore_validation)





# Create plots to see given player's prediction versus actual -------
filtered_player = boxscore_validation[boxscore_validation['PLAYER_NAME'] == 'Jaylen Brown']

sns.scatterplot(x=filtered_player['fantasy_points'], y=filtered_player['fantasy_point_pred'])


# create a line & scatter plot of actual versus predicted values
sns.lineplot(x='GAME_DATE_EST', y='fantasy_point_pred', data=filtered_player, label='Predicted')
sns.scatterplot(x='GAME_DATE_EST', y='fantasy_points', data=filtered_player, label='Actual')

