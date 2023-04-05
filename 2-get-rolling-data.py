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

# pull rolling data since last pulled
# nba api seems to bblock cloud servers 


def get_game_ids_pulled():
    
    s3_path = "s3://nbadk-model/game_stats"

    game_headers = wr.s3.read_parquet(
        path=s3_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

    game_ids_pulled = game_headers.GAME_ID.unique()
    latest_game_date = pd.to_datetime(game_headers.GAME_DATE_EST).dt.strftime('%Y-%m-%d').unique().max()

    return game_ids_pulled, latest_game_date


def get_recent_game_header_n_line_score(start_date, game_ids):

    game_header_w_standings_list = []
    team_game_line_score_list = []
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
            game_header_w_standings_list.append(game_header_w_standings)

            # each line rpresents a game-teamid
            team_game_line_score = scoreboard.line_score.get_data_frame()
            team_game_line_score_list.append(team_game_line_score)
        
        except Exception as e:
            error_dates_list.append(current_date)
            print(f'error {current_date}')

        current_date += timedelta(days=1)
        print(current_date)

        time.sleep(1.1)

    game_header_w_standings_complete_df = pd.concat(game_header_w_standings_list)
    game_header_w_standings_complete_df = game_header_w_standings_complete_df[game_header_w_standings_complete_df['LIVE_PERIOD'] >= 4] # filter for games that are completed
    
    team_game_line_score_complete_df = pd.concat(team_game_line_score_list) 
    team_game_line_score_complete_df = team_game_line_score_complete_df[team_game_line_score_complete_df['GAME_ID'].isin(game_header_w_standings_complete_df['GAME_ID'])]

    game_header_w_standings_filtered = game_header_w_standings_complete_df[~game_header_w_standings_complete_df['GAME_ID'].isin(game_ids)]
    team_game_line_score_filtered = team_game_line_score_complete_df[~team_game_line_score_complete_df['GAME_ID'].isin(game_ids)]


    game_header_w_standings_filtered.reset_index(inplace=True)
    team_game_line_score_filtered.reset_index(inplace=True)

    game_ids = game_header_w_standings_filtered.GAME_ID.unique()
    game_ids_df = pd.DataFrame(game_ids, columns=['game_id'])

    print(f'game ids to pull {len(game_ids)}')
    print('Writing Game Header data to S3...........')

    wr.s3.to_parquet(
        df=game_header_w_standings_filtered,
        path=f's3://nbadk-model/game_stats/game_header/game_header_{end_date_string}.parquet'
    )

    wr.s3.to_parquet(
        df=team_game_line_score_filtered,
        path=f's3://nbadk-model/team_stats/game_line_score/game_line_score_{end_date_string}.parquet'
    )

    wr.s3.to_parquet(
        df=game_ids_df,
        path=f's3://nbadk-model/api_ids/game_ids/game_id_{end_date_string}.parquet'
    )
    
    return game_ids



# GET BOXSCORE STATS ------------------------------------------------------------
## these are just additional boxscore stat metrics (e.g. offesnive rating, usage percentge) 
def get_boxscore_advanced(game_ids):
    
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

    print(player_boxscore_advanced_stats_df.shape)
    
    player_ids = player_boxscore_advanced_stats_df.PLAYER_ID.unique()
    player_ids_df = pd.DataFrame(player_ids, columns=['player_id'])

    print('Writing Boxscore Advanced to S3..........')

    wr.s3.to_parquet(
        df=player_boxscore_advanced_stats_df,
        path=f's3://nbadk-model/player_stats/boxscore_advanced/player_boxscore_advanced_stats_{today_string}.parquet'
    )

    wr.s3.to_parquet(
        df=team_boxscore_stats_advanced_df,
        path=f's3://nbadk-model/team_stats/boxscore_advanced/team_boxscore_advanced_stats_{today_string}.parquet'
    )



def get_boxscore_traditional(game_ids):
    
    print('Starting Boxscore Traditional....................')
    today = date.today()
    today_string = today.strftime('%Y-%m-%d')

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

    print('Writing Boxscore Traditional to S3......................')

    wr.s3.to_parquet(
            df=boxscore_traditional_player_df,
            path=f's3://nbadk-model/player_stats/boxscore_traditional/boxscore_traditional_player_{today_string}.parquet'
        )

    wr.s3.to_parquet(
        df=boxscore_traditional_team_df,
        path=f's3://nbadk-model/team_stats/boxscore_traditional/boxscore_traditional_team_{today_string}.parquet'
        )




game_ids_already_pulled, last_date = get_game_ids_pulled()

game_ids_to_pull = get_recent_game_header_n_line_score(last_date, game_ids_already_pulled)

get_boxscore_advanced(game_ids_to_pull)

get_boxscore_traditional(game_ids_to_pull)

