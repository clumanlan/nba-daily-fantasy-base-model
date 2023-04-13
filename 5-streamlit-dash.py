import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import awswrangler as wr
from st_aggrid import AgGrid
from pulp import *
# Load and prep data -------------------------------------------------------
today = date.today().strftime('%Y-%m-%d')

# Read in predictions
pred_path = f's3://nbadk-model/predictions/base/linear_regression/'
player_pred = wr.s3.read_parquet(
        path=pred_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

# Read in draftkings salaries
dk_salaries_path = f's3://nbadk-model/draftkings/roster-salaries/DKSalaries_{today}.csv'
dk_salaries_today =  wr.s3.read_csv(
        path = dk_salaries_path,
        path_suffix = ".csv" ,
        use_threads =True
    )

dk_salaries_today = dk_salaries_today[['Roster Position', 'Name', 'ID', 'Salary', 'AvgPointsPerGame']]
dk_salaries_today.rename({'Name':'PLAYER_NAME'},axis=1, inplace=True)

# Read in player correlations
player_correlation_path = f's3://nbadk-model/processed/correlations/rolling/player/nba_player_correlations_{today}.parquet'
player_correlations = wr.s3.read_parquet(
        path=player_correlation_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

# Get today's predictions 
player_pred['GAME_DATE_EST'] = pd.to_datetime(player_pred['GAME_DATE_EST'])
latest_pred_date = player_pred['GAME_DATE_EST'].max()

player_pred_rel_cols = ['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'is_starter', 'fantasy_point_prediction',
                        'player_fantasy_points_rank_overall_lagged','fantasy_points_rank_overall_lagged_team_allowed_opposing']

player_pred_latest = player_pred[player_pred['GAME_DATE_EST']==latest_pred_date][player_pred_rel_cols]

player_pred_latest = pd.merge(player_pred_latest, dk_salaries_today, how='inner')
player_pred_latest['Roster Position'] = player_pred_latest['Roster Position'].astype(str)






st.set_page_config(page_title="Current Fantasy Lineup", layout="wide") 



# Optimization ---------------------------------------

# Filter dataframe

player_ids_to_remove = []
player_pred_latest_filtered =  player_pred_latest[~player_pred_latest['PLAYER_NAME'].isin(player_ids_to_remove)]

player_ids = player_pred_latest_filtered['PLAYER_ID'].unique()
player_pred_latest_filtered = player_pred_latest_filtered.set_index('PLAYER_ID')

# Correlation filtered for players playing
player_correlations_filtered = player_correlations[player_correlations['corr']>=0.5]
player_correlations_filtered = player_correlations_filtered[(player_correlations_filtered['MAIN_PLAYER_ID'].isin(player_ids)) &  (player_correlations_filtered['OTHER_PLAYER_ID'].isin(player_ids))]

player_correlation_pairs = player_correlations_filtered[['MAIN_PLAYER_ID', 'OTHER_PLAYER_ID']].values.tolist()



lineup_lp_prob = LpProblem('dk_lineup', LpMaximize)

# create decision variables
player_vars = LpVariable.dicts('player', player_ids, cat = 'Binary')

# objective 
lineup_lp_prob += lpSum([player_pred_latest_filtered['fantasy_point_prediction'][i]*player_vars[i] for i in player_ids]), 'Objective Function'

# total salary constraints
lineup_lp_prob += lpSum([player_pred_latest_filtered['Salary'][i]*player_vars[i] for i in player_ids]) <= 50000

# roster 10 player constraints
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids]) == 8

for pair in player_correlation_pairs:
    lineup_lp_prob += player_vars[pair[0]] + player_vars[pair[1]] >= 2 # at least one player from each correlated pair must be included in the lineup

# lineup constraints 
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'G' in player_pred_latest_filtered['Roster Position'][i] ]) >= 3 
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'G' in player_pred_latest_filtered['Roster Position'][i] ]) <= 4

lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'F' in player_pred_latest_filtered['Roster Position'][i] ]) >= 3
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'F' in player_pred_latest_filtered['Roster Position'][i] ]) <= 4
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'PG' in player_pred_latest_filtered['Roster Position'][i]]) >= 1
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'SG' in player_pred_latest_filtered['Roster Position'][i]]) >= 1
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'SF' in player_pred_latest_filtered['Roster Position'][i]]) >= 1
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'PF' in player_pred_latest_filtered['Roster Position'][i]]) >= 1
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'C' in player_pred_latest_filtered['Roster Position'][i]]) >= 1
lineup_lp_prob += lpSum([player_vars[i] for i in player_ids if 'C' in player_pred_latest_filtered['Roster Position'][i]]) <= 2

lineup_lp_prob.solve()

player_ids=[]
player_lineup_status=[]
for val in player_vars:
        
    player_ids.append(val)
    player_lineup_status.append(player_vars[val].varValue)

final_lineup = pd.DataFrame({"PLAYER_ID":player_ids,"Status":player_lineup_status})
final_lineup = final_lineup[final_lineup["Status"]==1].join(player_pred_latest_filtered, on = 'PLAYER_ID', how='left')

final_lineup = final_lineup.drop(columns=['PLAYER_ID', 'Status', 'ID'], axis=1)
final_lineup.fantasy_point_prediction.sum()

st.title("Current Fantasy Lineup")
AgGrid(final_lineup)



PLAYER CORRELATIONS
TEAM CORRELATIONS 


