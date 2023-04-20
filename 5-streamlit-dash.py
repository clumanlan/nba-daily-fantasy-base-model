import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import awswrangler as wr
from st_aggrid import AgGrid
from pulp import *
from st_aggrid.shared import GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import jinja2
import plotly_express as px
import urllib3
from nba_api.stats.static import teams
from PIL import Image
import boto3

st.set_page_config(page_title="Current Fantasy Lineup", layout="wide") 

def load_and_prep_data():
    # Load teams lookup data
    teams_lookup = pd.DataFrame(teams.get_teams())
    teams_lookup = teams_lookup[['id', 'full_name']]
    teams_lookup = teams_lookup.rename(columns={'id':'TEAM_ID', 'full_name':'team_full_name'})
    teams_lookup['TEAM_ID'] = teams_lookup['TEAM_ID'].astype(str)

    # Get today's date
    today = date.today().strftime('%Y-%m-%d')

    # Read in processed data 
    processed_path = f's3://nbadk-model/processed/base_model_processed/nba_base_processed_{today}.parquet'
    base_model_processed = wr.s3.read_parquet(
        path=processed_path,
        use_threads=True
    )
    base_model_processed = base_model_processed[['PLAYER_NAME', 'PLAYER_ID', 'GAME_ID', 'GAME_DATE_EST_x', 'is_starter', 'fantasy_points']]
    base_model_processed = base_model_processed.drop_duplicates()

    # Read in predictions
    pred_path = f's3://nbadk-model/predictions/base/linear_regression/nba_base_lr_pred_{today}.parquet'
    player_pred = wr.s3.read_parquet(
        path=pred_path,
        use_threads=True
    )
    player_pred_rel_cols = ['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'is_starter', 'fantasy_point_prediction', 'player_fantasy_points_rank_overall_lagged']
    player_pred_latest = player_pred[player_pred['GAME_DATE_EST'] == today][player_pred_rel_cols]
    
    # Read in draftkings salaries
    dk_salaries_path = f's3://nbadk-model/draftkings/roster-salaries/DKSalaries_{today}.csv'
    dk_salaries_today =  wr.s3.read_csv(
        path=dk_salaries_path,
        path_suffix=".csv",
        use_threads=True
    )
    dk_salaries_today = dk_salaries_today[['Roster Position', 'Name', 'ID', 'Salary', 'AvgPointsPerGame']]
    dk_salaries_today.rename({'Name':'PLAYER_NAME'}, axis=1, inplace=True)

    # Merge predictions with salaries
    player_pred_latest = pd.merge(player_pred_latest, dk_salaries_today, how='inner')
    player_pred_latest['Roster Position'] = player_pred_latest['Roster Position'].astype(str)

    # Read in player correlations
    player_correlation_path = f's3://nbadk-model/processed/correlations/rolling/player/nba_player_correlations_{today}.parquet'
    player_correlations = wr.s3.read_parquet(
        path=player_correlation_path,
        path_suffix=".parquet",
        use_threads=True
    )

    return teams_lookup, base_model_processed, player_pred_latest, player_correlations, dk_salaries_today

teams_lookup, base_model_processed, player_pred_latest, player_correlations, dk_salaries_today= load_and_prep_data()


# Main Page -------------------------------------------------------------------
tab1, tab2 = st.tabs(["Current Lineup", "Player Correlations"])

# Tab 1 --------------------------------------------------------------------
with tab1: 

    col1, col2, col3 = st.columns([1, 7, 1])

    with col1:
        st.write(' ')

    with col2:
            
        st.title("Current Fantasy Lineup")
        st.write('Generate a fantasy lineup based on both individual player fantasy point predictions and the correlation between players\' fantasy points.', font="italic")
        st.markdown("""
        - <b>Step 1:</b> Exclude players and the lineup optimization will automatically rerun (often times there are player's ruled out right before gametime)
        - <b>Step 2:</b> Select a player and visualize the distribution of a player's fantasy points and fantasy point trend over time.
        """, unsafe_allow_html=True)

        # Optimization ---------------------------------------

        # Filter dataframe
        
        player_ids_to_remove = st.multiselect('Select Players to Remove:', player_pred_latest['PLAYER_NAME'])
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

        # ensure at least on player from correlated pairs is included 
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


        final_lineup = final_lineup.drop(columns=['PLAYER_ID', 'Status', 'ID', 'Roster Position'], axis=1)
        final_lineup['fantasy_point_prediction'] =  final_lineup['fantasy_point_prediction'].round()
        final_lineup.fantasy_point_prediction.sum()

        col_defs = [{'headerName':'Player', 'field':'PLAYER_NAME','checkboxSelection': True,  'width': 180},
                        {'headerName':'Position', 'field':'POSITION', 'width':90},
                        {'headerName':'Starter', 'field':'is_starter', 'width':90},
                        {'headerName':'Fantasy Point Prediction', 'field':'fantasy_point_prediction', 'width':180, 'headerStyle':{'textAlign': 'center'}, 'cellStyle': {'textAlign': 'center'}},
                        {'headerName':'Avg Points Per Game', 'field':'AvgPointsPerGame', 'width':180, 'headerStyle':{'textAlign': 'center'}, 'cellStyle': {'textAlign': 'center'}}, 
                        {'headerName':'Player FP Rank Overall', 'field':'player_fantasy_points_rank_overall_lagged', 'width':180, 'headerStyle':{'textAlign': 'center'}, 'cellStyle': {'textAlign': 'center'}},
                        {'headerName':'Salary', 'field':'Salary', 'width':90}]


        gb = GridOptionsBuilder.from_dataframe(final_lineup)
        gridOptions = gb.build()
        gridOptions['columnDefs'] = col_defs

        final_lineup_grid = AgGrid(final_lineup, 
                    gridOptions=gridOptions, 
                    enable_enterprise_modules=True, 
                    allow_unsafe_jscode=True, 
                    update_mode=GridUpdateMode.SELECTION_CHANGED)

    selected_rows = final_lineup_grid["selected_rows"]
    
    with col3:
        st.write(' ')

    if len(selected_rows) != 0:
            
        selected_rows_df = pd.DataFrame(selected_rows)
        player_selected = selected_rows_df['PLAYER_NAME']
        player_selected_var = player_selected.iloc[0]
    
    else: 
        player_selected = ['Alex Caruso']
        player_selected_var = 'Alex Caruso'
        

    filtered_base_df = base_model_processed[base_model_processed["PLAYER_NAME"].isin(player_selected)][['PLAYER_NAME','GAME_DATE_EST_x','fantasy_points']]
        
    st.write(f"<h1 style='text-align: center; font-weight: bold; font-size: 24px;'>{player_selected_var}</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_a = px.histogram(filtered_base_df["fantasy_points"], nbins=30, opacity=0.5)
        fig_a.update_layout(
            title='Player Fantsay Point Actual Histogram',
            xaxis_title='Fantasy Poimts',
            yaxis_title='Observations',
            yaxis=dict(dtick=1)
        )

        st.plotly_chart(fig_a)
        
    with col2:
        fig_b = px.scatter(filtered_base_df, x="GAME_DATE_EST_x", y="fantasy_points",trendline="ols", opacity=0.5)
        fig_b.update_layout(
            title='Player Fantasy Point Actual Over Time',
            xaxis_title='Game Date',
            yaxis_title='Fantasy Points'
        )

        st.plotly_chart(fig_b)
    
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.write(' ')
    
    with col2:
  
        st.info("Read more about how the model works and see the code on my [Github](https://github.com/clumanlan/nba-daily-fantasy-base-model).", icon="ℹ️")

    with col3:
        st.write(' ')


    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.write(' ')
    
    with col2:
        st.write(' ')
  
    with col3:
        
        img = Image.open("images/bball-fp-logo.jpg")
        # Resize image
        width, height = img.size
        new_height = 100
        new_width = int(new_height * (width / height))
        resized_img = img.resize((new_width, new_height))
        st.image(resized_img)

            

with tab2:
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.write(' ')

    with col2:
        st.write('Top Player Correlations')

        #PLAYER CORRELATIONS
        player_lookup_table = base_model_processed[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
        player_correlations_filtered_wnames = pd.merge(player_correlations_filtered, player_lookup_table,
                                                    left_on='MAIN_PLAYER_ID', right_on='PLAYER_ID', how='left')
        player_correlations_filtered_wnames =  pd.merge(player_correlations_filtered_wnames, player_lookup_table, 
                                                    left_on='OTHER_PLAYER_ID', right_on='PLAYER_ID', how='left')

        player_correlations_filtered_wnames = player_correlations_filtered_wnames.drop(['PLAYER_ID_x', 'PLAYER_ID_y'], axis=1)
        player_correlations_filtered_wnames = player_correlations_filtered_wnames.rename(columns={'PLAYER_NAME_x':'MAIN_PLAYER_NAME', 'PLAYER_NAME_y':'OTHER_PLAYER_NAME', 'TEAM_ID_x':'TEAM_ID'})

        player_correlations_filtered_wnames = pd.merge(player_correlations_filtered_wnames, teams_lookup,  how='left')
        player_correlations_filtered_wnames['players'] = player_correlations_filtered_wnames[['MAIN_PLAYER_NAME', 'OTHER_PLAYER_NAME']].apply(lambda x: '_'.join(sorted(x)), axis=1)
        player_correlations_filtered_wnames = player_correlations_filtered_wnames.drop_duplicates(subset='players')
        player_correlations_filtered_wnames = player_correlations_filtered_wnames[['team_full_name','MAIN_PLAYER_NAME', 'OTHER_PLAYER_NAME', 'corr']]
        player_correlations_filtered_wnames['corr'] = player_correlations_filtered_wnames['corr'].round(2)
        player_correlations_filtered_wnames = player_correlations_filtered_wnames.sort_values('corr', ascending=False)

        jscode = JsCode("""
            function(params) {
                if (params.data.corr >= 0.7) {
                    return {
                'color': 'grey',
                'backgroundColor': '#abf7b1'
                            }
                        }
                    };
                    """)

                    
        col_defs_corr = [{'headerName':'Team', 'field':'team_full_name', 'width': 120},
                        {'headerName':'Player', 'field':'MAIN_PLAYER_NAME', 'width':180},
                        {'headerName':'Player', 'field':'OTHER_PLAYER_NAME', 'width':180},
                        {'headerName':'Fantasy Point Correlation', 'field':'corr', 'width':180}]


        gb_corr = GridOptionsBuilder.from_dataframe(player_correlations_filtered_wnames)
        gridOptions_corr = gb_corr.build()
        gridOptions_corr['getRowStyle'] = jscode
        gridOptions_corr['columnDefs'] = col_defs_corr

        grid_response = AgGrid(
            player_correlations_filtered_wnames,
            gridOptions=gridOptions_corr,
            allow_unsafe_jscode=True,
                )
        
        with col3:
            st.write(' ')

#TEAM CORRELATIONS 


