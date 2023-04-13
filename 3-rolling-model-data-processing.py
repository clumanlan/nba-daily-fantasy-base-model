
from nba_api.stats.endpoints import BoxScoreAdvancedV2, LeagueGameLog, TeamPlayerDashboard, BoxScoreDefensive, commonplayerinfo, CommonTeamRoster
import time as time
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
import awswrangler as wr
from datetime import date
from nba_api.stats.endpoints import ScoreboardV2
from bs4 import BeautifulSoup
import requests
from nba_api.stats.static import teams

# Define relevant features -----------------------------------


rel_num_cols = ['seconds_played', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A','FTM', 'FTA',
        'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS',
        'E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING',
        'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO',
        'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
        'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE']


# This function converts minutes played into total seconds --------------------
def get_sec(time_str):
    """Get seconds from time."""
    if ':' in time_str:
        if '.' in time_str:
            time_str = time_str.replace('.000000', '')
            m, s = time_str.split(':')
            time_sec = int(m) *60 + int(s)
        else:
            m, s = time_str.split(':')
            time_sec = int(m)*60 + int(s)
    
    if ':' not in time_str:
        if time_str == 'None':
            time_sec = 0
        else: 
            time_sec = int(time_str)*60

    return time_sec


# This function calculates fantasy points based on draftkings 
def calculate_fantasy_points(df):
        return df['PTS'] + df['FG3M']*0.5 + df['REB']*1.25 + df['AST']*1.5 + df['STL']*2 + df['BLK']*2 - df['TO']*0.5

# Basic function to pull stats
def create_aggregate_rolling_functions(window_num = 20, window_min = 1):
        ## aggregate rolling functions to create summary stats
        f_min = lambda x: x.rolling(window=window_num, min_periods=window_min).min() 
        f_max = lambda x: x.rolling(window=window_num, min_periods=window_min).max()
        f_mean = lambda x: x.rolling(window=window_num, min_periods=window_min).mean()
        f_std = lambda x: x.rolling(window=window_num, min_periods=window_min).std()
        f_sum = lambda x: x.rolling(window=window_num, min_periods=window_min).sum()

        return f_min, f_max, f_mean, f_std, f_sum

def get_current_season_games() -> tuple:
    """
    Retrieves current season game headers, processes them, and creates a long table
    to flag teams as home or away.

    Parameters:
    game_stats_path (str): S3 path to game stats parquet file.

    Returns:
    tuple: A tuple containing the following:
        - game_headers_df_processed_filtered (pd.DataFrame): Processed game headers for current season.
        - game_home_away (pd.DataFrame): Long table to flag teams as home or away for each game.
        - current_season_game_ids (np.ndarray): Array of unique game IDs for current season.
    """
    game_stats_path = "s3://nbadk-model/game_stats"

    game_headers_df = wr.s3.read_parquet(
        path=game_stats_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

    current_season = game_headers_df.SEASON.max()
    current_season_game_ids = game_headers_df[game_headers_df['SEASON']==current_season]['GAME_ID'].unique()


    game_headers_df_processed = (game_headers_df[game_headers_df['SEASON'] == current_season]
        .assign(
            gametype_string = game_headers_df.GAME_ID.str[:3],
            game_type = lambda x: np.where(x.gametype_string == '001', 'Pre-Season',
                np.where(x.gametype_string == '002', 'Regular Season',
                np.where(x.gametype_string == '003', 'All Star',
                np.where(x.gametype_string == '004', 'Post Season',
                np.where(x.gametype_string == '005', 'Play-In Tournament', 'unknown'))))),
            GAME_ID = game_headers_df['GAME_ID'].astype(str),
            GAME_DATE_EST = pd.to_datetime(game_headers_df['GAME_DATE_EST'])

        )
    )

    game_headers_df_processed.drop_duplicates(subset=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'], inplace=True)

    rel_cols = ['GAME_ID', 'game_type', 'SEASON', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS', 'HOME_TEAM_LOSSES']

    game_headers_df_processed_filtered = game_headers_df_processed[rel_cols]
    game_headers_df_processed_filtered = game_headers_df_processed_filtered.drop_duplicates()

    # Create long table in order to flag teams that are home or away
    game_home_away = game_headers_df_processed[['GAME_ID','HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    game_home_away = pd.melt(game_home_away, id_vars='GAME_ID', value_name='TEAM_ID', var_name='home_away')
    game_home_away['home_away'] = game_home_away['home_away'].apply(lambda x: 'home' if x == 'HOME_TEAM_ID' else 'away')

    return game_headers_df_processed_filtered, game_home_away, current_season_game_ids


def get_player_dfs(rel_game_ids:list) -> tuple:
    """
    Get dataframes for player information, traditional box score stats, and advanced box score stats 
    for a given set of game IDs.

    Args:
        rel_game_ids (list): List of relevant game IDs to filter box score dataframes by.

    Returns:
        tuple: A tuple of three pandas dataframes: player_info_df, boxscore_trad_player_df, and boxscore_adv_player_df.
    """

    # Read in player information data
    player_info_path = "s3://nbadk-model/player_info"
    player_info_df = wr.s3.read_parquet(
        path=player_info_path,
        path_suffix=".parquet",
        use_threads=True
    )

    # Clean player information data and rename columns
    player_info_df = player_info_df[['PERSON_ID', 'HEIGHT', 'POSITION']].drop_duplicates()
    player_info_df = player_info_df.rename({'PERSON_ID': 'PLAYER_ID'}, axis=1)

    # Read in traditional box score data for players and filter by relevant game IDs
    boxscore_trad_player_path = "s3://nbadk-model/player_stats/boxscore_traditional/"
    boxscore_trad_player_df = wr.s3.read_parquet(
        path=boxscore_trad_player_path,
        path_suffix=".parquet",
        use_threads=True
    )
    boxscore_trad_player_df['GAME_ID'] = boxscore_trad_player_df['GAME_ID'].astype(str)
    boxscore_trad_player_df = boxscore_trad_player_df[boxscore_trad_player_df['GAME_ID'].isin(rel_game_ids)]

    # Read in advanced box score data for players and filter by relevant game IDs
    boxscore_adv_player_path = "s3://nbadk-model/player_stats/boxscore_advanced/"
    boxscore_adv_player_df = wr.s3.read_parquet(
        path=boxscore_adv_player_path,
        path_suffix=".parquet",
        use_threads=True
    )
    boxscore_adv_player_df = boxscore_adv_player_df.drop_duplicates(subset=['GAME_ID','PLAYER_ID'])
    boxscore_adv_player_df = boxscore_adv_player_df[boxscore_adv_player_df['GAME_ID'].isin(rel_game_ids)]

    return player_info_df, boxscore_trad_player_df, boxscore_adv_player_df



def get_team_level_dfs(rel_game_ids:list) -> tuple:
    """
    Retrieve team level dataframes for the given game IDs.

    Args:
    rel_game_ids (list): A list of game IDs to filter the dataframes by.

    Returns:
    tuple: A tuple of two pandas dataframes, the first containing traditional team stats and the second containing advanced team stats.
    """

    # Read in traditional boxscore team stats
    boxscore_trad_team_path = "s3://nbadk-model/team_stats/boxscore_traditional/"

    boxscore_trad_team_df = wr.s3.read_parquet(
        path=boxscore_trad_team_path,
        path_suffix=".parquet",
        use_threads=True
    )

    # Convert GAME_ID to string and filter by rel_game_ids
    boxscore_trad_team_df['GAME_ID'] = boxscore_trad_team_df['GAME_ID'].astype(str)
    boxscore_trad_team_df = boxscore_trad_team_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])
    boxscore_trad_team_df = boxscore_trad_team_df[boxscore_trad_team_df['GAME_ID'].isin(rel_game_ids)]

    # Read in advanced boxscore team stats
    boxscore_adv_team_path = "s3://nbadk-model/team_stats/boxscore_advanced/"

    boxscore_adv_team_df = wr.s3.read_parquet(
        path=boxscore_adv_team_path,
        path_suffix=".parquet",
        use_threads=True
    )

    # Drop duplicates and filter by rel_game_ids
    boxscore_adv_team_df = boxscore_adv_team_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])
    boxscore_adv_team_df = boxscore_adv_team_df[boxscore_adv_team_df['GAME_ID'].isin(rel_game_ids)]

    return boxscore_trad_team_df, boxscore_adv_team_df


def create_player_level_boxscore(player_info_df:pd.DataFrame,
                                 boxscore_trad_player_df:pd.DataFrame,
                                 boxscore_adv_player_df:pd.DataFrame, 
                                 game_headers_df:pd.DataFrame):
    """
    Creates a complete player-level boxscore by merging player info, traditional player stats, advanced player stats, 
    and game headers dataframes.

    Parameters:
        player_info_df (pandas.DataFrame): Dataframe containing player information, with columns: 'PLAYER_ID', 
            'HEIGHT', and 'POSITION'.
        boxscore_trad_player_df (pandas.DataFrame): Dataframe containing traditional player stats, with columns: 
            'GAME_ID', 'TEAM_ID', 'PLAYER_ID', and other stats.
        boxscore_adv_player_df (pandas.DataFrame): Dataframe containing advanced player stats, with columns: 'GAME_ID', 
            'TEAM_ID', 'PLAYER_ID', and other stats.
        game_headers_df (pandas.DataFrame): Dataframe containing game headers information, with columns: 'GAME_ID', 
            'game_type', 'SEASON', and 'GAME_DATE_EST'.

    Returns:
        pandas.DataFrame: A complete player-level boxscore dataframe.
    """
    
    # Merge the player info dataframe to add player positions
    boxscore_complete_player = pd.merge(boxscore_trad_player_df, player_info_df, on='PLAYER_ID', how='left')

    # Merge the advanced player dataframe using multiple columns and suffixes
    boxscore_complete_player = pd.merge(
        boxscore_complete_player,
        boxscore_adv_player_df,
        on=['GAME_ID', 'PLAYER_ID', 'TEAM_ID'],
        how='left',
        suffixes=['', '_adv']
    )

    # Merge the filtered game headers dataframe to add game information
    game_info_df = game_headers_df[['GAME_ID', 'game_type', 'SEASON', 'GAME_DATE_EST']]
    boxscore_complete_player = pd.merge(boxscore_complete_player, game_info_df, on='GAME_ID', how='left')

    # Filter out pre-season and all-star-games 
    boxscore_complete_player = boxscore_complete_player[~boxscore_complete_player['game_type'].isin(['Pre-Season', 'All Star'])]

    return boxscore_complete_player


def create_team_level_boxscore(boxscore_trad_team_df:pd.DataFrame, 
                               boxscore_adv_team_df:pd.DataFrame,
                               game_home_away:pd.DataFrame,
                               game_headers_df:pd.DataFrame):
    """
    Creates a complete boxscore for team-level statistics by merging traditional and advanced stats and game information.

    Args:
        boxscore_trad_team_df (pd.DataFrame): Dataframe containing traditional boxscore stats for each team.
        boxscore_adv_team_df (pd.DataFrame): Dataframe containing advanced boxscore stats for each team.
        game_home_away (pd.DataFrame): Dataframe containing home and away team for each game.
        game_headers_df (pd.DataFrame): Dataframe containing game information such as game type, season, and date.

    Returns:
        pd.DataFrame: A complete boxscore for team-level statistics with merged traditional and advanced stats and game information.
    """

    # Merge traditional and advanced stats dataframes on game id and team id
    boxscore_complete_team = pd.merge(
        boxscore_trad_team_df,
        boxscore_adv_team_df,
        on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=['', '_adv']
    )

    # Subset game information dataframe to only include necessary columns
    game_info_df = game_headers_df[['GAME_ID', 'game_type', 'SEASON', 'GAME_DATE_EST']]

    # Merge home and away team information with boxscore dataframe on game id and team id
    boxscore_complete_team = pd.merge(boxscore_complete_team, game_home_away, how='left', on=['GAME_ID', 'TEAM_ID'])

    # Merge game information with boxscore dataframe on game id
    boxscore_complete_team = pd.merge(boxscore_complete_team, game_info_df, on='GAME_ID', how='left')

    # Filter out pre-season and all-star games from the boxscore dataframe
    boxscore_complete_team = boxscore_complete_team[~boxscore_complete_team['game_type'].isin(['Pre-Season', 'All Star'])]

    return boxscore_complete_team


def reindex_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindexes a DataFrame by date to fill in missing dates, and forwards fills the NaN values.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing dates.
    
    Returns:
        pd.DataFrame: The reindexed DataFrame with missing dates filled.
    """
    
    # Define range of dates based on minimum and maximum dates in DataFrame
    dates = pd.date_range(df.GAME_DATE_EST.min(), df.GAME_DATE_EST.max())
    
    # Reindex the DataFrame by date and forward fill missing values
    return df.reindex(dates).ffill()


def lag_player_values(df: pd.DataFrame, rel_num_cols: list) -> pd.DataFrame:
    """
    Creates lagged player statistics by shifting numerical columns in a DataFrame based on specified groups.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing player statistics.
        rel_num_cols (List[str]): A list of numerical column names in the DataFrame to be shifted.
    
    Returns:
        pd.DataFrame: The DataFrame with lagged player statistics added as new columns.
    """
    
    # Sort the DataFrame by game date
    df = df.sort_values(['GAME_DATE_EST'])
    
    # Create lagged columns by grouping on specified columns and shifting numerical columns by 1
    df = (
        df.assign(**{
            f'player_{col}_lagged': df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'], group_keys=False)[col].shift(1)
            for col in rel_num_cols})
        .reset_index(drop=True)
    )
    
    # Get list of lagged column names
    lagged_cols =  list(df.columns[df.columns.str.endswith('_lagged')])
    
    # Select columns to keep in output
    df = df[['PLAYER_ID', 'GAME_ID'] + lagged_cols]

    return df


def create_lagged_player_rolling_agg_stats(df: pd.DataFrame, rel_num_cols: list) -> pd.DataFrame:
    """
    Creates lagged player statistics by shifting numerical columns in a DataFrame based on specified groups and aggregating
    with multiple summary statistics.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing player statistics.
        rel_num_cols (List[str]): A list of numerical column names in the DataFrame to be shifted and aggregated.
    
    Returns:
        pd.DataFrame: The DataFrame with lagged player statistics added as new columns.
    """
    
    # Sort the DataFrame by game date
    df = df.sort_values(['GAME_DATE_EST'])
    
    # Create lagged columns by grouping on specified columns and shifting numerical columns by 1
    df = (
        df.assign(**{
            f'player_{col}_lagged': df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'], group_keys=False)[col].shift(1)
            for col in rel_num_cols})
        .reset_index(drop=True)
    )
    
    # Drop original numerical columns
    df.drop(rel_num_cols, axis=1, inplace=True)
    
    # Define list of summary statistics and names
    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    function_name = ['min', 'max', 'mean', 'std', 'sum']
    
    # Iterate over lagged columns and summary statistics to create new columns
    for col in df.columns[df.columns.str.endswith('_lagged')]:
        for i in range(len(function_list)):
            df[(col + '_%s' % function_name[i])] = df.sort_values(['GAME_DATE_EST']).groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'], group_keys=False)[col].apply(function_list[i])
    
    return df


def create_player_level_features(boxscore_complete_player: pd.DataFrame, rel_num_cols: list) -> pd.DataFrame:
    """
    Creates player level features based on the boxscore data for each game they played.
    
    Parameters:
    boxscore_complete_player (pd.DataFrame): Dataframe containing the boxscore data for all players and games.
    rel_num_cols (list): List of columns containing relevant numerical values for lagging.
    
    Returns:
    pd.DataFrame: Dataframe with player level features for each game they played.
    """
    boxscore_complete_player['seconds_played'] = boxscore_complete_player['MIN'].apply(get_sec)

    boxscore_complete_player_processed = (boxscore_complete_player
        .assign(fantasy_points = calculate_fantasy_points(boxscore_complete_player),
            SEASON_ID = lambda x: x['SEASON'].astype('int'),
            Player_ID = lambda x: x['PLAYER_ID'].astype('str'),
            is_starter = np.where(boxscore_complete_player['START_POSITION']=="", False, True)
            )
        .sort_values(by=['GAME_DATE_EST'])
        .reset_index(drop=True)
        .drop(['MIN'], axis=1)
    )

    # Calculate rolling games played 
    rolling_count_games_played = boxscore_complete_player_processed.sort_values('GAME_DATE_EST').groupby(['PLAYER_ID','SEASON'])['GAME_ID'].rolling(window=100).count()
    boxscore_complete_player_processed['rolling_games_played'] = rolling_count_games_played.droplevel([0,1]).sort_index().reset_index(drop=True)

    # Filter out players that did not play
    boxscore_complete_player_processed = boxscore_complete_player_processed[boxscore_complete_player_processed['seconds_played'] > 0]

    # Create Player Ranking
    boxscore_complete_player_processed = boxscore_complete_player_processed.sort_values(['GAME_DATE_EST'])

    boxscore_complete_player_processed['fantasy_points_lagged'] = (
        boxscore_complete_player_processed
        .groupby(['PLAYER_ID'], 
                group_keys=False)['fantasy_points']
                .shift(1)
    )

    rolling_avg = boxscore_complete_player_processed.groupby(['PLAYER_ID', 'SEASON_ID'])['fantasy_points_lagged'].rolling(window=100, min_periods=1).mean()
    rolling_avg = rolling_avg.rename('fantasy_points_lagged_mean')
    rolling_avg = rolling_avg.reset_index().set_index('level_2').sort_index()
    boxscore_complete_player_processed['fantasy_points_lagged_mean'] = rolling_avg['fantasy_points_lagged_mean']

    # Create rolling player ranking
    player_season_calendar_list = []

    for season in boxscore_complete_player_processed['SEASON_ID'].unique():
        df = boxscore_complete_player_processed[boxscore_complete_player_processed['SEASON_ID']==season]
        df.index = pd.DatetimeIndex(df.GAME_DATE_EST)

        df = df.groupby('PLAYER_ID').apply(reindex_by_date).reset_index(0, drop=True)
        player_season_calendar_list.append(df)

        print(season)


    player_season_ranking_calendar_df = pd.concat(player_season_calendar_list)
    player_season_ranking_calendar_df = player_season_ranking_calendar_df[['PLAYER_ID', 'POSITION', 'fantasy_points_lagged_mean', 'SEASON_ID']].reset_index(names='calendar_date')
    player_season_ranking_calendar_df = player_season_ranking_calendar_df.dropna(subset='fantasy_points_lagged_mean')
    player_season_ranking_calendar_df['fantasy_points_rank_overall'] = player_season_ranking_calendar_df.groupby('calendar_date')['fantasy_points_lagged_mean'].rank(ascending=False)
    player_season_ranking_calendar_df = player_season_ranking_calendar_df.drop(['POSITION', 'SEASON_ID', 'fantasy_points_lagged_mean'], axis=1)

    # Add ranking to the player boxscore
    boxscore_complete_player_processed = pd.merge(boxscore_complete_player_processed, player_season_ranking_calendar_df, left_on= ['GAME_DATE_EST', 'PLAYER_ID'], right_on =['calendar_date', 'PLAYER_ID'], how='left')
    boxscore_complete_player_processed = boxscore_complete_player_processed.drop('fantasy_points_lagged',axis=1)


    rel_num_cols_add = ['fantasy_points_rank_overall', 'rolling_games_played']

    add_player_lagged_stats = lag_player_values(boxscore_complete_player_processed, rel_num_cols_add)


    # Function to lag all stats and calculate basic statistics of them from above

    f_min, f_max, f_mean, f_std, f_sum = create_aggregate_rolling_functions()


    ## rel no lag columns ------------
    rel_cols_player_no_lag = ['GAME_ID', 'SEASON_ID', 'GAME_DATE_EST', 'is_starter', 'PLAYER_ID', 'PLAYER_NAME', 'START_POSITION', 'POSITION', 'TEAM_ID', 'fantasy_points']

    f_min, f_max, f_mean, f_std, f_sum = create_aggregate_rolling_functions()
    boxscore_complete_player_processed = boxscore_complete_player_processed[rel_cols_player_no_lag + rel_num_cols]
    boxscore_complete_player_processed = create_lagged_player_rolling_agg_stats(boxscore_complete_player_processed, rel_num_cols)

    # Add in additional stats created earlier (games_played, fantasy_points_rank_overall)
    boxscore_complete_player_processed = pd.merge(boxscore_complete_player_processed, add_player_lagged_stats, on=['PLAYER_ID', 'GAME_ID'], how='left')

    return boxscore_complete_player_processed



# Create Team Level Features ------------------------------------------------------------------------ 

def create_lagged_team_stats(df: pd.DataFrame, rel_num_cols:list) -> pd.DataFrame:
    """
    Create lagged team stats and basic stats from them, same as player plus additional fantasy points.
    
    Args:
    - df: pandas.DataFrame, input dataframe with team stats.
    - rel_num_cols: list, relevant numerical columns to lag.
    
    Returns:
    - pandas.DataFrame, the input dataframe with added lagged columns and basic stats.
    """
    # Sort the dataframe by game date in ascending order
    df = df.sort_values(['GAME_DATE_EST'])

    # Create lagged columns for relevant numerical columns by grouping them by team and season ID, then shifting them by 1 game
    df = (
        df.assign(**{
        f'team_{col}_lagged': df.groupby(['TEAM_ID', 'SEASON_ID'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
        .reset_index(drop=True)
    )

    # Remove the original relevant numerical columns, as they are no longer needed
    df.drop(rel_num_cols, axis=1, inplace=True)

    # Define a list of functions to apply to the lagged columns to calculate basic stats
    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    # Define a list of names for each basic stat
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    # For each lagged column in the dataframe
    for col in df.columns[df.columns.str.endswith('_lagged')]:
        print(col)
        # For each basic stat in the list of functions to apply
        for i in range(len(function_list)):
            # Apply the function to the lagged column by grouping by team and season ID, then add the basic stat column to the dataframe
            df[(col + '_%s' % function_name[i])] = df.sort_values(['GAME_DATE_EST']).groupby(['TEAM_ID', 'SEASON_ID'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])

    # Return the updated dataframe with lagged columns and basic stats
    return df



def create_team_level_features(boxscore_complete_team:pd.DataFrame, rel_num_cols:list) -> pd.DataFrame:
    """
    Creates lagged team statistics and basic statistics from them, as well as additional fantasy points.

    Args:
    df (pandas.DataFrame): DataFrame containing the team statistics.
    rel_num_cols (list): List of relevant numerical columns.

    Returns:
    pandas.DataFrame: DataFrame containing the lagged team statistics, basic statistics, and additional fantasy points.
    """

    boxscore_complete_team['seconds_played'] = boxscore_complete_team['MIN'].apply(get_sec)

    boxscore_complete_team_processed = (boxscore_complete_team
        .assign(fantasy_points_team = calculate_fantasy_points(boxscore_complete_team),
            SEASON_ID = lambda x: x['SEASON'].astype('int'),
            TEAM_ID = lambda x: x['TEAM_ID'].astype('str')
            )
        .drop('MIN', axis=1)
        )


    # Get fantasy points allowed from other team
    fantasy_points_allowed = boxscore_complete_team_processed[['GAME_ID', 'TEAM_ID', 'fantasy_points_team']]
    boxscore_complete_team_processed = pd.merge(boxscore_complete_team_processed, fantasy_points_allowed, on='GAME_ID', suffixes=['', '_opposing'], how='left')
    boxscore_complete_team_processed = boxscore_complete_team_processed[boxscore_complete_team_processed['TEAM_ID'].astype(str) != boxscore_complete_team_processed['TEAM_ID_opposing'].astype(str)]

    # Lag team fantasy points and opposing team fantasy points
    boxscore_complete_team_processed = boxscore_complete_team_processed.sort_values(['GAME_DATE_EST'])

    team_group_shift = boxscore_complete_team_processed.groupby(['TEAM_ID'])[['fantasy_points_team', 'fantasy_points_team_opposing']].shift(1)
    team_group_shift.columns = ['fantasy_points_team_lagged', 'fantasy_points_team_allowed_lagged']

    boxscore_complete_team_processed = pd.concat([boxscore_complete_team_processed, team_group_shift], axis=1)

    rolling_avg_team = boxscore_complete_team_processed.groupby(['TEAM_ID', 'SEASON_ID'])[['fantasy_points_team', 'fantasy_points_team_opposing']].rolling(window=100, min_periods=1).mean()
    rolling_avg_team = rolling_avg_team.rename(columns={'fantasy_points_team': 'fantasy_points_team_lagged_mean', 'fantasy_points_team_opposing': 'fantasy_points_team_opposing_lagged_mean'})
    rolling_avg_team = rolling_avg_team.reset_index().set_index('level_2').sort_index()

    boxscore_complete_team_processed[['fantasy_points_team_lagged_mean', 'fantasy_points_team_opposing_lagged_mean']] = rolling_avg_team[['fantasy_points_team_lagged_mean', 'fantasy_points_team_opposing_lagged_mean']]


    # Create team ranking of fantasy points scored & allowed
    boxscore_complete_team_processed[boxscore_complete_team_processed.index.duplicated()]

    team_season_calendar_list = []

    for season in boxscore_complete_team_processed['SEASON_ID'].unique():
        df = boxscore_complete_team_processed[boxscore_complete_team_processed['SEASON_ID']==season]
        df.index = pd.DatetimeIndex(df.GAME_DATE_EST)

        df = df.groupby('TEAM_ID').apply(reindex_by_date).reset_index(0, drop=True)
        team_season_calendar_list.append(df)

        print(season)


    team_season_lagged_ranking_df = pd.concat(team_season_calendar_list)

    team_season_lagged_ranking_df = team_season_lagged_ranking_df[['TEAM_ID', 'SEASON_ID', 'fantasy_points_team_lagged_mean', 'fantasy_points_team_opposing_lagged_mean']].reset_index(names='calendar_date')
    team_season_lagged_ranking_df = team_season_lagged_ranking_df.dropna(subset='fantasy_points_team_lagged_mean')

    # Rank team's fantasy points scored
    team_season_lagged_ranking_df['fantasy_points_rank_overall_lagged_team'] = team_season_lagged_ranking_df.groupby('calendar_date')['fantasy_points_team_lagged_mean'].rank(ascending=False)

    # Rank how many points they give up to their opponent with the team giving up most points being #1
    team_season_lagged_ranking_df['fantasy_points_rank_overall_lagged_team_allowed'] = team_season_lagged_ranking_df.groupby('calendar_date')['fantasy_points_team_opposing_lagged_mean'].rank() 
    team_season_lagged_ranking_df = team_season_lagged_ranking_df[['calendar_date', 'TEAM_ID', 'fantasy_points_rank_overall_lagged_team', 'fantasy_points_rank_overall_lagged_team_allowed']]


    rel_team_cols_no_lag = ['GAME_ID', 'TEAM_ID', 'GAME_DATE_EST', 'SEASON_ID',  'home_away', 'game_type']
    rel_num_cols_team = rel_num_cols + ['fantasy_points_team']

    boxscore_complete_team_processed = boxscore_complete_team_processed[rel_team_cols_no_lag + rel_num_cols_team]
    boxscore_complete_team_processed = create_lagged_team_stats(boxscore_complete_team_processed, rel_num_cols_team)


    # Add ranking to the team boxscore
    boxscore_complete_team_processed = pd.merge(boxscore_complete_team_processed, team_season_lagged_ranking_df, left_on= ['GAME_DATE_EST', 'TEAM_ID'], right_on =['calendar_date', 'TEAM_ID'], how='left')
    boxscore_complete_team_processed = boxscore_complete_team_processed.drop(['SEASON_ID', 'SEASON_ID'], axis=1)
    boxscore_complete_team_processed = pd.merge(boxscore_complete_team_processed, boxscore_complete_team_processed, on='GAME_ID', suffixes=['','_opposing'], how='left')
    
    boxscore_complete_team_processed = boxscore_complete_team_processed[boxscore_complete_team_processed['TEAM_ID'] != boxscore_complete_team_processed['TEAM_ID_opposing']]
    
    return boxscore_complete_team_processed


def pull_todays_games():
    """
    This function pulls today's NBA games from the NBA API and returns a dataframe with the game information.

    Parameters:
    date_param (str): a string in the format of 'YYYY-MM-DD' representing the date for which to pull games

    Returns:
    todays_games (pandas.DataFrame): a dataframe containing the following columns for each game:
        - GAME_ID: a unique ID for each game
        - SEASON: the season in which the game is being played
        - GAME_DATE_EST: the date of the game
        - home_away: a string indicating whether the team is playing at home or away
        - TEAM_ID: the ID of the team playing in the game
        - team_abb: the abbreviation of the team playing in the game
    """
    # Get today's date
    today = date.today().strftime('%Y-%m-%d')
    
    # Get scoreboard for today's games
    scoreboard = ScoreboardV2(game_date=today, league_id='00')

    # Get game header and series standings data
    game_header = scoreboard.game_header.get_data_frame()
    series_standings = scoreboard.series_standings.get_data_frame()

    # Merge home and away team data into a single column
    game_home_away = game_header[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    game_home_away = pd.melt(game_home_away, id_vars='GAME_ID', value_name='TEAM_ID', var_name='home_away')
    game_home_away['home_away'] = np.where(game_home_away['home_away'] == 'HOME_TEAM_ID', 'home', 'away')
    
    # Get team ID and abbreviation data
    teams_json = teams.get_teams()
    teams_df = pd.DataFrame(teams_json)[['id', 'abbreviation']]
    teams_df.columns = ['TEAM_ID', 'team_abb']
    
    # Merge team abbreviation data into game_home_away data
    game_home_away = game_home_away.merge(teams_df, how='left')

    # Select relevant columns and rename columns for today's games data
    rel_cols = ['GAME_ID', 'SEASON', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']
    todays_game_header = game_header[rel_cols].drop_duplicates()
    todays_game_header = todays_game_header.rename({'HOME_TEAM_ID': 'home', 'VISITOR_TEAM_ID': 'away'}, axis=1)

    # Merge home and away team data into a single column for today's games data
    todays_games = pd.melt(todays_game_header, id_vars=['GAME_ID', 'SEASON', 'GAME_DATE_EST'],
                value_vars=['home', 'away'],
                var_name='home_away', value_name='TEAM_ID')

    # Convert team ID data to string type
    todays_games['TEAM_ID'] = todays_games['TEAM_ID'].astype(str)

    return todays_games


def pull_todays_roster(todays_teams):
    """
    Pulls the roster of today's teams and their starting lineup information (if available).

    Args:
    - todays_teams (list of str): A list of team IDs (as strings) for which to pull roster information.

    Returns:
    - todays_roster_complete (pandas DataFrame): A DataFrame with roster information for all players on today's teams,
      as well as a boolean flag indicating whether they are a starter for today's game (if available).
      Columns include: PLAYER_ID, PLAYER_NAME, POSITION, and TEAM_ID.
    """

    todays_roster_complete = []

    for id in todays_teams:

        # Get roster information for each team
        roster = CommonTeamRoster(team_id=id).common_team_roster.get_data_frame()
        rel_roster_cols = ['PLAYER_ID', 'PLAYER', 'POSITION', 'TeamID']
        roster = roster[rel_roster_cols]

        todays_roster_complete.append(roster)
        print(f'processing team id {id}')
        time.sleep(0.67)

    todays_roster_complete = pd.DataFrame(pd.concat(todays_roster_complete))
    todays_roster_complete.rename({'PLAYER':'PLAYER_NAME', 'TeamID':'TEAM_ID'}, axis=1, inplace=True)

    # Get starting lineup information from rotowire.com
    url = "https://www.rotowire.com/basketball/nba-lineups.php"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    
    lineups = soup.find_all(class_='is-pct-play-100') # get todays starting lineups
    positions = [x.find('div').text for x in lineups]
    names = [x.find('a')['title'] for x in lineups]
    teams = sum([[x.text] * 5 for x in soup.find_all(class_='lineup__abbr')], [])

    df_starters = pd.DataFrame(zip(names, teams, positions))
    df_starters.columns = ['PLAYER_NAME', 'team_abb', 'position']
    df_starters['starter'] = True
    df_starters.drop(['position', 'team_abb'], axis=1, inplace=True)

    lineups_injury = soup.find_all(class_='is-pct-play-0') # get todays injured
    names_injury = [x.find('a')['title'] for x in lineups_injury]
    
    todays_roster_complete = todays_roster_complete.merge(df_starters, how='left')
    todays_roster_complete = todays_roster_complete[~todays_roster_complete['PLAYER_NAME'].isin(names_injury)]
    todays_roster_complete['starter'] = todays_roster_complete['starter'].fillna(False)

    return todays_roster_complete





f_min, f_max, f_mean, f_std, f_sum = create_aggregate_rolling_functions()

game_headers_df, game_home_away, current_season_game_ids = get_current_season_games()

# Team level processing --------------------------------

# Get team level stats
boxscore_trad_team_df, boxscore_adv_team_df = get_team_level_dfs(current_season_game_ids)
boxscore_complete_team = create_team_level_boxscore(boxscore_trad_team_df, boxscore_adv_team_df, game_home_away, game_headers_df)

# Get latest record for each team
boxscore_complete_team = create_team_level_features(boxscore_complete_team, rel_num_cols)
boxscore_team_latest = boxscore_complete_team.loc[boxscore_complete_team.groupby('TEAM_ID')['GAME_DATE_EST'].idxmax()]
boxscore_team_latest = boxscore_team_latest.drop(['GAME_ID',  'home_away'], axis=1)


# Get teams playing today
todays_game_headers = pull_todays_games()
todays_game_headers = pd.merge(todays_game_headers, boxscore_team_latest, on='TEAM_ID', how='left')


# Player level processing -----------------------------
# Get player level stats
player_info_df, boxscore_trad_player_df, boxscore_adv_player_df = get_player_dfs(current_season_game_ids)
boxscore_complete_player = create_player_level_boxscore(player_info_df, boxscore_trad_player_df, boxscore_adv_player_df, game_headers_df)
boxscore_complete_player_processed = create_player_level_features(boxscore_complete_player, rel_num_cols)
boxscore_complete_player_processed['TEAM_ID'] = boxscore_complete_player_processed['TEAM_ID'].astype(str)

# Get latest player level stat for each player
boxscore_player_latest = (
    boxscore_complete_player_processed
    .sort_values(['GAME_DATE_EST'])
    .loc[boxscore_complete_player_processed.groupby('PLAYER_ID')['GAME_DATE_EST'].idxmax()]
    .drop(['GAME_ID', 'GAME_DATE_EST', 'POSITION', 'PLAYER_NAME', 'TEAM_ID'], axis=1)
)

# Get players playing today
todays_player_roster = pull_todays_roster(todays_game_headers['TEAM_ID'])
todays_player_roster = pd.merge(todays_player_roster, boxscore_player_latest, on='PLAYER_ID', how='left')

# Merge player and team dataframes
combined_player_team_boxsccore_current_season = (
    pd.merge(
        boxscore_complete_player_processed,
        boxscore_complete_team,
        on=['TEAM_ID', 'GAME_ID'],
        how='left'
    )
)

# Save combined data to S3
today = date.today().strftime('%Y-%m-%d')
wr.s3.to_parquet(
    df=combined_player_team_boxsccore_current_season,
    path=f's3://nbadk-model/processed/base_model_processed/nba_base_processed_{today}.parquet'
)

# Merge latest player and team dataframes
todays_player_roster['TEAM_ID'] = todays_player_roster['TEAM_ID'].astype(str)
combined_player_team_boxscore_latest = (
    pd.merge(
        todays_player_roster,
        todays_game_headers,
        how='left',
        on=['TEAM_ID']
    )
    .dropna()
    .rename(columns={'GAME_DATE_EST_x': 'GAME_DATE_EST'})
)


# Load Linear Regression model -----------------------------------------------------------
lr_run_id = '29f299a29533425bb969722eceb5929d'
model_uri = f"projects/nba-daily-fantasy-base-model/base_model/lr_base/"
lr_base = mlflow.sklearn.load_model(model_uri)


# Predict player's fantasy points
todays_fantasy_point_pred = lr_base.predict(combined_player_team_boxscore_latest)
combined_player_team_boxscore_latest['fantasy_point_prediction'] = todays_fantasy_point_pred


wr.s3.to_parquet(
        df=combined_player_team_boxscore_latest,
        path=f's3://nbadk-model/predictions/base/linear_regression/nba_base_lr_pred_{today}.parquet'
        )


