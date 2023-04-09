import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import awswrangler as wr
import time as time
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, ScoreboardV2, BoxScoreAdvancedV2, BoxScoreTraditionalV2
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players
import awswrangler as wr
from typing import List, Tuple

def get_game_ids_pulled(s3_path: str) -> Tuple[pd.Series, str]:
    """
    Retrieves the unique game IDs and latest game date from game_stats data previously pulled.
    
    Parameters:
    s3_path (str): S3 path to the game_stats data.
    
    Returns:
    Tuple[pd.Series, str]: A tuple containing a pandas Series of unique game IDs pulled and the latest game date in the data.
    """

    s3_path = "s3://nbadk-model/game_stats"

    game_headers = wr.s3.read_parquet(
        path=s3_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

    game_ids_pulled = game_headers.GAME_ID.unique()
    latest_game_date = pd.to_datetime(game_headers.GAME_DATE_EST).dt.strftime('%Y-%m-%d').unique().max()

    return game_ids_pulled, latest_game_date

def get_game_data(start_date:str) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], List[date]]:
    """
    Fetches game data from ScoreboardV2 API for each date from `start_date` to today's date.
    :param start_date: A string representing the start date in the format YYYY-MM-DD.
    :return: A tuple containing a list of tuples of Game Header and Team Game Line Scores dataframes and a list of dates where an error occurred.
    """

    game_data = []
    error_dates_list = []

    start_date = datetime.strptime(start_date, '%Y-%m-%d').date() - timedelta(days=5)

    end_date = date.today()
    end_date_string = end_date.strftime('%Y-%m-%d')

    current_date = start_date

    while current_date <= end_date:
        try:
            scoreboard = ScoreboardV2(game_date=current_date, league_id='00')

            game_header = scoreboard.game_header.get_data_frame()

            series_standings = scoreboard.series_standings.get_data_frame()
            series_standings.drop(['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE_EST'], axis=1, inplace=True)

            game_header_w_standings = game_header.merge(series_standings, on='GAME_ID')

            # each line rpresents a game-teamid
            team_game_line_score = scoreboard.line_score.get_data_frame()
            game_data.append(game_header_w_standings, team_game_line_score)
        
        except Exception as e:
            error_dates_list.append(current_date)
            print(f'error {current_date}')

        current_date += timedelta(days=1)
        print(current_date)

        time.sleep(1.1)

    return game_data, error_dates_list


def filter_game_data(game_data: List[Tuple[pd.DataFrame, pd.DataFrame]], game_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Filters the game data by removing games that are not yet completed and have already been processed.

    :param game_data: A list of tuples containing two dataframes representing game data.
    :param game_ids: A list of integers representing game IDs that have already been processed.

    :return: A tuple containing two dataframes representing the filtered game data and a list of Game IDS to pull
    """
    game_header_w_standings_list = []
    team_game_line_score_list = []

    for game_header_w_standings, team_game_line_score in game_data:
        if game_header_w_standings['LIVE_PERIOD'].iloc[-1] >= 4 and game_header_w_standings['GAME_ID'].iloc[-1] not in game_ids:
            game_header_w_standings_list.append(game_header_w_standings)
            team_game_line_score_list.append(team_game_line_score)

    game_header_w_standings_complete_df = pd.concat(game_header_w_standings_list)
    team_game_line_score_complete_df = pd.concat(team_game_line_score_list)
    game_ids = game_header_w_standings_complete_df.GAME_ID.unique()

    print(f'game ids to pull {len(game_ids)}')


    return game_header_w_standings_complete_df, team_game_line_score_complete_df, game_ids





def write_game_data_to_s3(game_header_w_standings_filtered: pd.DataFrame, team_game_line_score_filtered: pd.DataFrame, output_date: str = None) -> None:
    """
    Writes the filtered game data to S3 in Parquet format.

    :param game_header_w_standings_filtered: A dataframe representing the filtered game header data.
    :param team_game_line_score_filtered: A dataframe representing the filtered team game line score data.
    :param output_date: A string representing the output date in YYYY-MM-DD format. Default is today's date.

    """

    if output_date is None:
        output_date = datetime.today().strftime('%Y-%m-%d')

    print('Writing Game Header data to S3...........')

    wr.s3.to_parquet(
        df=game_header_w_standings_filtered,
        path=f's3://nbadk-model/game_stats/game_header/game_header_{end_date_string}.parquet'
    )

    wr.s3.to_parquet(
        df=team_game_line_score_filtered,
        path=f's3://nbadk-model/team_stats/game_line_score/game_line_score_{end_date_string}.parquet'
    )


def get_boxscore_advanced(game_ids:list) ->  Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Retrieves box score advanced statistics (e.g. PACE) for a list of game ids.

    Args:
        game_ids (List[str]): List of game ids to retrieve box score advanced statistics for.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]: A tuple containing two dataframes - player_boxscore_advanced_stats_df
        and team_boxscore_stats_advanced_df

    """
    print('Starting Boxscore Advanced...................')

    today = date.today()
    today_string = today.strftime('%Y-%m-%d')

    player_boxscore_stats_list = []
    team_boxscore_stats_list = []
    error_game_id_list = []

    game_len = len(game_ids)
    loop_place = 0

    for game_id in game_ids:
        print(f'Starting {game_id}')

        try:
            boxscore_stats_adv = BoxScoreAdvancedV2(game_id=game_id)

            player_boxscore_stats = boxscore_stats_adv.player_stats.get_data_frame()
            team_boxscore_stats = boxscore_stats_adv.team_stats.get_data_frame()

            player_boxscore_stats_list.append(player_boxscore_stats)
            team_boxscore_stats_list.append(team_boxscore_stats)

            print(f'success {game_id}')
        
        except Exception as e:
            error_game_id_list.append(game_id)

            print(f'error {game_id}')
        
        loop_place += 1
        print(f'{(loop_place/game_len)*100} % complete')
        time.sleep(1)
    
    player_boxscore_advanced_stats_df = pd.concat(player_boxscore_stats_list)
    team_boxscore_stats_advanced_df = pd.concat(team_boxscore_stats_list)

    return player_boxscore_advanced_stats_df, team_boxscore_stats_advanced_df


def write_boxscore_advanced_to_s3(player_boxscore_advanced_stats_df: pd.DataFrame, 
                                  team_boxscore_stats_advanced_df: pd.DataFrame, 
                                  today_string: str) -> None:
    """
    Writes the boxscore advanced data to S3 in Parquet format.

    :param player_boxscore_advanced_stats_df: A dataframe representing the player boxscore advanced stats data.
    :param team_boxscore_stats_advanced_df: A dataframe representing the team boxscore advanced stats data.

        """
    today = date.today()
    today_string = today.strftime('%Y-%m-%d')

    print('Writing Boxscore Advanced to S3..........')

    wr.s3.to_parquet(
        df=player_boxscore_advanced_stats_df,
        path=f's3://nbadk-model/player_stats/boxscore_advanced/player_boxscore_advanced_stats_{today_string}.parquet'
    )

    wr.s3.to_parquet(
        df=team_boxscore_stats_advanced_df,
        path=f's3://nbadk-model/team_stats/boxscore_advanced/team_boxscore_advanced_stats_{today_string}.parquet'
    )

    print('Finished writing Boxscore Advanced Team & Player to S3.')
    return None


def get_boxscore_traditional(game_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve traditional boxscore data for given game IDs.

    Parameters:
    -----------
    game_ids : list of str
        List of game IDs for which boxscore data is to be retrieved.

    Returns:
    --------
    Tuple of two dataframes: player-level boxscore data and team-level boxscore data.
    """

    print('Starting Boxscore Traditional....................')

    boxscore_trad_player_list = []
    boxscore_trad_team_list = []
    boxscore_trad_error_list = []
    game_len = game_ids.shape[0]
    loop_place = 0

    for game_id in game_ids:
        try:
            boxscore_trad = BoxScoreTraditionalV2(game_id=game_id)

            boxscore_trad_player = boxscore_trad.player_stats.get_data_frame()
            boxscore_trad_team = boxscore_trad.team_stats.get_data_frame()

            boxscore_trad_player_list.append(boxscore_trad_player)
            boxscore_trad_team_list.append(boxscore_trad_team)

            print(game_id)
        
        except Exception as e:
            boxscore_trad_error_list.append(game_id)

            print(f'error {game_id}')
        
        time.sleep(.60)
        loop_place += 1
        print(f'{(loop_place/game_len)*100} % complete')

    boxscore_traditional_player_df = pd.concat(boxscore_trad_player_list)
    boxscore_traditional_team_df = pd.concat(boxscore_trad_team_list)

    boxscore_traditional_player_df['MIN'] = boxscore_traditional_player_df['MIN'].astype(str)
    boxscore_traditional_team_df['MIN'] = boxscore_traditional_team_df['MIN'].astype(str)

    return boxscore_traditional_player_df, boxscore_traditional_team_df


def write_boxscore_traditional_to_s3(boxscore_traditional_player_df: pd.DataFrame, boxscore_traditional_team_df: pd.DataFrame) -> None:
    """
    Writes boxscore traditional stats for players and teams to S3 in parquet format with today's date appended to the filename.

    Parameters:
        boxscore_traditional_player_df (pd.DataFrame): DataFrame of player boxscore traditional stats
        boxscore_traditional_team_df (pd.DataFrame): DataFrame of team boxscore traditional stats
    """
    today = date.today()
    today_string = today.strftime('%Y-%m-%d')

    print('Writing Boxscore Traditional to S3......................')

    wr.s3.to_parquet(
            df=boxscore_traditional_player_df,
            path=f's3://nbadk-model/player_stats/boxscore_traditional/boxscore_traditional_player_{today_string}.parquet'
        )

    wr.s3.to_parquet(
        df=boxscore_traditional_team_df,
        path=f's3://nbadk-model/team_stats/boxscore_traditional/boxscore_traditional_team_{today_string}.parquet'
        )
    
    print('Finished writing Boxscore Traditional Team & Player to S3.')

    return None



game_ids_already_pulled, last_date = get_game_ids_pulled()

game_ids_to_pull = get_recent_game_header_n_line_score(last_date, game_ids_already_pulled)

get_boxscore_advanced(game_ids_to_pull)

get_boxscore_traditional(game_ids_to_pull)

