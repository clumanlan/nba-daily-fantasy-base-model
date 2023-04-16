import pandas as pd
import awswrangler as wr
from datetime import date



def read_player_team_base_model_processed_from_s3() -> pd.DataFrame:

    today = date.today().strftime('%Y-%m-%d')
    boxscore_path = f's3://nbadk-model/processed/base_model_processed/'

    player_team_base_model_processed = wr.s3.read_parquet(
            path=boxscore_path,
            path_suffix = ".parquet" ,
            use_threads =True
        )
    
    return player_team_base_model_processed

def get_processed_current_season(base_model_processed: pd.DataFrame) -> pd.DataFrame:
   
   """
    Processes player and team fantasy point correlations and saves them as parquet files in S3.

    Args:
        base_model_processed (pd.DataFrame): DataFrame containing processed data for all seasons.
        boxscore_path (str): S3 path where the base model processed data is located.
        output_path (str): S3 path where the output data will be saved.

    Returns:
        None
    """
   latest_processed_season = base_model_processed.SEASON_ID.max()
   current_season_box = base_model_processed[base_model_processed['SEASON_ID'] == latest_processed_season]
   current_season_box = current_season_box.drop_duplicates(subset=['TEAM_ID', 'GAME_ID', 'PLAYER_NAME', 'PLAYER_ID', 'fantasy_points'])
   
   return current_season_box

def calculate_player_correlations(base_model_processed: pd.DataFrame, output_path: str) -> None:

    # Filter for relevant columns to get player fantasy point correlations  
    player_fp = base_model_processed[['TEAM_ID', 'GAME_ID', 'PLAYER_NAME', 'PLAYER_ID', 'fantasy_points']]

    # Drop duplicates as a result of processing each day
    player_fp = player_fp.drop_duplicates() 

    team_player_corr_dict = {}
    teams = player_fp['TEAM_ID'].unique()

    for team in teams:
        player_fp_filtered = player_fp[player_fp['TEAM_ID'] == team]

        player_fp_filtered = player_fp_filtered.pivot(
                index='GAME_ID',
                columns='PLAYER_ID',
                values='fantasy_points').fillna(0)

        corr = player_fp_filtered.corr()

        corr = corr.rename_axis(None).rename_axis(None, axis=1)
        corr = corr.stack().reset_index()
        corr.columns = ['MAIN_PLAYER_ID','MAIN_PLAYER_NAME', 'OTHER_PLAYER_ID', 'OTHER_PLAYER_NAME', 'corr']

        corr = pd.DataFrame(corr)

        team_player_corr_dict[team] = corr

    team_player_corr_df = pd.concat(team_player_corr_dict).reset_index().rename(columns={'level_0': 'TEAM_ID', 'level_1':'index'})
    team_player_corr_df = team_player_corr_df.drop(['index'],axis=1)

    # Filter out correlations that are a player against themselves
    team_player_corr_df = team_player_corr_df[team_player_corr_df['corr'] != 1]

    today = date.today().strftime('%Y-%m-%d')
    wr.s3.to_parquet(
            df=team_player_corr_df,
            path=output_path.format('player', today)
            )
    
    return None
    
def calculate_team_correlations(base_model_processed: pd.DataFrame, output_path: str) -> None:

    # Filter for relevant columns to get team fantasy point correlations  
    team_fp = (
        base_model_processed
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'fantasy_points':'sum'})
    )
    team_fp = team_fp.reset_index()

    team_fp = team_fp.pivot(
            index='GAME_ID',
            columns='TEAM_ID',
            values='fantasy_points').fillna(0)

    team_corr = team_fp.corr()

    team_corr = team_corr.rename_axis(None).rename_axis(None, axis=1)
    team_corr = team_corr.stack().reset_index()
    team_corr.columns = ['MAIN_TEAM_ID', 'OTHER_TEAM_ID', 'corr']

    team_corr = pd.DataFrame(team_corr)

    # Filter out correlations that are a team against itself
    team_corr = team_corr[team_corr['corr']!=1]

    today = date.today().strftime('%Y-%m-%d')

    wr.s3.to_parquet(
            df=team_corr,
            path=output_path.format('team', today)
            )
    
    return None

def main():

    player_team_base_model_processed = read_player_team_base_model_processed_from_s3()

    today = date.today().strftime('%Y-%m-%d')
    calculate_player_correlations(player_team_base_model_processed, 
                                  output_path=f's3://nbadk-model/processed/correlations/rolling/player/nba_player_correlations_{today}.parquet')
    calculate_team_correlations(player_team_base_model_processed, 
                                output_path=f's3://nbadk-model/processed/correlations/rolling/team/nba_team_correlations_{today}.parquet')



main()
