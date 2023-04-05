
import pandas as pd
import awswrangler as wr
from datetime import date

today = date.today().strftime('%Y-%m-%d')
boxscore_path = f's3://nbadk-model/processed/base-rolling/nba_base_processed_{today}.parquet'

player_team_box_current_season = wr.s3.read_parquet(
        path=boxscore_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )


# Player Correlations --------------------------------

player_fp = player_team_box_current_season[['TEAM_ID', 'GAME_ID', 'PLAYER_NAME', 'fantasy_points']]

team_corr_dfs_dict = {}
teams = player_fp['TEAM_ID'].unique()

for team in teams:

    player_fp_filtered = player_fp[player_fp['TEAM_ID'] == team]

    player_fp_filtered = player_fp_filtered.pivot(
            index='GAME_ID',
            columns='PLAYER_NAME',
            values='fantasy_points').fillna(0)
    
    corr = player_fp_filtered.corr()
    

    corr = corr.rename_axis(None).rename_axis(None, axis=1)
    corr = corr.stack().reset_index()
    corr.columns = ['PLAYER_NAME_A', 'PLAYER_NAME_B', 'corr']

    corr = pd.DataFrame(corr)

    team_corr_dfs_dict[team] = corr

    print(team)


team_corr_df = pd.concat(team_corr_dfs_dict).reset_index().rename(columns={'level_0': 'TEAM_ID', 'level_1':'index'})
team_corr_df = team_corr_df.drop(['index'],axis=1)

# Filter out correlations that are a player against themselves
team_corr_df = team_corr_df[team_corr_df['corr'] != 1]


#team correlations 
team_corr_df_threshold_pos = team_corr_df[team_corr_df['corr'] > 0.4]
team_corr_df_threshold_pos[['PLAYER_NAME_A', 'PLAYER_NAME_B']].values.tolist()



# TEAM CORRELATIONS ---------------------------------------------------------

player_fp = player_team_box_current_season[['TEAM_ID', 'GAME_ID', 'PLAYER_NAME', 'fantasy_points_team']]
player_team_box_current_season.columns