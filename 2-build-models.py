import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import BoxScoreAdvancedV2, LeagueGameLog, TeamPlayerDashboard, BoxScoreDefensive, commonplayerinfo
import time as time
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import time
import awswrangler as wr
from category_encoders import TargetEncoder
from sklearn.linear_model import LinearRegression

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

# READ IN DATA & QUICK PROCESSING -------------------------------------------------------

player_info_path = "s3://nbadk-model/player_info"


player_info_df = wr.s3.read_parquet(
    path=player_info_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

player_info_df = player_info_df[['PERSON_ID', 'HEIGHT', 'POSITION']].drop_duplicates()
player_info_df = player_info_df.rename({'PERSON_ID': 'PLAYER_ID'}, axis=1)


game_stats_path = "s3://nbadk-model/game_stats"

game_headers_df = wr.s3.read_parquet(
    path=game_stats_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

game_headers_df_processed = (game_headers_df
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


boxscore_trad_player_path = "s3://nbadk-model/player_stats/boxscore_traditional/"

boxscore_trad_player_df = wr.s3.read_parquet(
    path=boxscore_trad_player_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

boxscore_trad_player_df['GAME_ID'] = boxscore_trad_player_df['GAME_ID'].astype(str)

boxscore_trad_team_path = "s3://nbadk-model/team_stats/boxscore_traditional/"

boxscore_trad_team_df = wr.s3.read_parquet(
    path=boxscore_trad_team_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

boxscore_trad_team_df['GAME_ID'] = boxscore_trad_team_df['GAME_ID'].astype(str)
boxscore_trad_team_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'], inplace=True)

boxscore_adv_player_path = "s3://nbadk-model/player_stats/boxscore_advanced/"

boxscore_adv_player_df = wr.s3.read_parquet(
    path=boxscore_adv_player_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

boxscore_adv_player_df = boxscore_adv_player_df.drop_duplicates(subset=['GAME_ID','PLAYER_ID'])

boxscore_adv_team_path = "s3://nbadk-model/team_stats/boxscore_advanced/"

boxscore_adv_team_df = wr.s3.read_parquet(
    path=boxscore_adv_team_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

boxscore_adv_team_df = boxscore_adv_team_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])

# create long table in order to flag teams that are home or away
game_home_away = game_headers_df_processed[['GAME_ID','HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
game_home_away = pd.melt(game_home_away, id_vars='GAME_ID', value_name='TEAM_ID', var_name='home_away')
game_home_away['home_away'] = game_home_away['home_away'].apply(lambda x: 'home' if x == 'HOME_TEAM_ID' else 'away')


# JOIN TABLES TO CREATE 2 DFS @ PLAYER AND TEAM LEVEL -----------------------------------------------------------

# Player Level DF -----------

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
game_info_df = game_headers_df_processed_filtered[['GAME_ID', 'game_type', 'SEASON', 'GAME_DATE_EST']]
boxscore_complete_player = pd.merge(boxscore_complete_player, game_info_df, on='GAME_ID', how='left')

# Filter out pre-season and all-star-games 
boxscore_complete_player = boxscore_complete_player[~boxscore_complete_player['game_type'].isin(['Pre-Season', 'All Star'])]

# Team Level DF ---------------

boxscore_complete_team = pd.merge(
    boxscore_trad_team_df,
    boxscore_adv_team_df,
    on=['GAME_ID', 'TEAM_ID'],
    how='left',
    suffixes=['', '_adv']
)

boxscore_complete_team = pd.merge(boxscore_complete_team, game_home_away, how='left', on=['GAME_ID', 'TEAM_ID'])
boxscore_complete_team = pd.merge(boxscore_complete_team, game_info_df, on='GAME_ID', how='left')
boxscore_complete_team = boxscore_complete_team[~boxscore_complete_player['game_type'].isin(['Pre-Season', 'All Star'])]




# player ids missing from the player info dataframe

missing_player_ids = boxscore_complete_player[boxscore_complete_player['POSITION'].isnull()][['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
boxscore_complete_player[boxscore_complete_player['POSITION'].isnull()][['PLAYER_ID', 'PLAYER_NAME', 'GAME_ID','GAME_DATE_EST', 'MIN']]


# DEFINE COLUMNS USED  --------------------------------------------------------------------------------
## base columns ---------
rel_id_cols_player = ['GAME_ID', 'PLAYER_ID', 'PLAYER_NAME', 'START_POSITION', 'POSITION', 'TEAM_ID']
rel_box_base_id_cols_team = ['GAME_ID', 'TEAM_ID']

## num columns -----------
rel_num_cols = [ 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A','FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS',
       'E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING',
       'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO',
       'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
       'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE']


# CREATE PLAYER LEVEL FEATURES -------------------------------------------------------------------------
boxscore_complete_player = boxscore_complete_player[rel_id_cols_player + rel_num_cols]

# this function calculates fantasy points based on draftkings structure
def calculate_fantasy_points(df):
        return df['PTS'] + df['FG3M']*0.5 + df['REB']*1.25 + df['AST']*1.5 + df['STL']*2 + df['BLK']*2 - df['TO']*0.5


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
boxscore_complete_player_processed = boxscore_complete_player[boxscore_complete_player['seconds_played'] > 0]



rel_box_base_num_cols.remove('MIN')
rel_box_base_num_cols.append('seconds_played')






boxscore_trad_n_adv_team['seconds_played'] = boxscore_trad_n_adv_team['MIN'].apply(get_sec)
boxscore_trad_n_adv_team.drop(['MIN'], axis=1, inplace=True)




# Create Player Ranking

boxscore_complete_player_processed['fantasy_points_lagged'] = boxscore_complete_player_processed.sort_values(['GAME_DATE_EST']).groupby(['PLAYER_ID'], group_keys=False)['fantasy_points'].shift(1)


# Sort by GAME_DATE_EST
boxscore_complete_player_processed = boxscore_complete_player_processed.sort_values(['GAME_DATE_EST'])

# Compute rolling average of 'fantasy_points_lagged' by PLAYER_ID and SEASON_ID
rolling_avg = boxscore_complete_player_processed.groupby(['PLAYER_ID', 'SEASON_ID'])['fantasy_points_lagged'].rolling(window=100, min_periods=1).mean()

# Rename the rolling average column
rolling_avg = rolling_avg.rename('fantasy_points_lagged_mean')

# Set the index to 'level_2'
rolling_avg = rolling_avg.reset_index().set_index('level_2').sort_index()


# Assign the rolling average to 'fantasy_points_lagged_avg' column
boxscore_complete_player_processed['fantasy_points_lagged_avg'] = rolling_avg


boxscore_complete_player_processed['fantasy_points_lagged_avg'] = (
    boxscore_complete_player_processed
    .sort_values(['GAME_DATE_EST'])
    .groupby(['PLAYER_ID', 'SEASON_ID'])
    .rolling(window=100, min_periods=1)
    ['fantasy_points_lagged'].mean()
    .reset_index()
    .set_index('level_2')
    .sort_index()
    .rename(columns={'fantasy_points_lagged':'fantasy_points_lagged_mean'})
    ['fantasy_points_lagged_mean']
)




def reindex_by_date(df):
    dates = pd.date_range(df.GAME_DATE_EST.min(), df.GAME_DATE_EST.max())
    return df.reindex(dates).ffill()

player_season_calendar_list = []

for season in boxscore_complete_player_processed['SEASON_ID'].unique():
    df = boxscore_complete_player_processed[boxscore_complete_player_processed['SEASON_ID']==season]
    df.index = pd.DatetimeIndex(df.GAME_DATE_EST)

    df = df.groupby('PLAYER_ID').apply(reindex_by_date).reset_index(0, drop=True)
    player_season_calendar_list.append(df)

    print(season)

player_season_ranking_calendar_df = pd.concat(player_season_calendar_list)
player_season_ranking_calendar_df = player_season_ranking_calendar_df[['PLAYER_ID', 'POSITION', 'fantasy_points_lagged_avg', 'SEASON_ID']].reset_index(names='calendar_date')
player_season_ranking_calendar_df = player_season_ranking_calendar_df.dropna(subset='fantasy_points_lagged_avg')

player_season_ranking_calendar_df['fantasy_points_rank_overall'] = player_season_ranking_calendar_df.groupby('calendar_date')['fantasy_points_lagged_avg'].rank(ascending=False)
player_season_ranking_calendar_df['fantasy_points_rank_position'] = player_season_ranking_calendar_df.groupby(['calendar_date', 'POSITION'])['fantasy_points_lagged_avg'].rank(ascending=False)

player_season_ranking_calendar_df[player_season_ranking_calendar_df['calendar_date']=='2000-11-02'].sort_values(by=['fantasy_points_lagged_avg'], ascending=False)





















# Create Team Level Features ------------------------------------------------------------------------ 

# LAST DATE GAME PLAYED DAYS 

combined_df_reg_base_team = (combined_df_reg_base_team
    .assign(fantasy_points_team = calculate_fantasy_points(combined_df_reg_base_team),
        SEASON_ID = lambda x: x['SEASON'].astype('int'),
        TEAM_ID = lambda x: x['TEAM_ID'].astype('str')
        )
    )
combined_df_reg_base_team = combined_df_reg_base_team[combined_df_reg_base_team['game_type'] == 'Regular Season'].reset_index(drop=True)

boxscore_complete_player_processed[['PLAYER_ID', 'GAME_DATE_EST']].value_counts(sort=True)
boxscore_complete_player_processed = boxscore_complete_player_processed.drop(178262) # drop index where shawn marion pplayed twice


boxscore_complete_player_processed[(boxscore_complete_player_processed['PLAYER_ID']==1890) & (boxscore_complete_player_processed['GAME_DATE_EST'] =='2007-12-19')]




# CREATE PLAYER HIERARCHY ----------------------------------------------------

player_position = boxscore_trad_player_df[['PLAYER_ID', 'POSITION']].drop_duplicates()



!! THIS IS NEXT HOW DO WE DO IT I THINK JUST BY FANTASY POINTS
# group by player_id rolling day fantasy point average 


add_num_cols_player = ['games_played']

# 'game_type'].isin(['Regular Season','Post Season', 'Play-In Tournament'])



def create_aggregate_rolling_functions(window_num = 20, window_min = 1):
    ## aggregate rolling functions to create summary stats
    f_min = lambda x: x.rolling(window=window_num, min_periods=window_min).min() 
    f_max = lambda x: x.rolling(window=window_num, min_periods=window_min).max()
    f_mean = lambda x: x.rolling(window=window_num, min_periods=window_min).mean()
    f_std = lambda x: x.rolling(window=window_num, min_periods=window_min).std()
    f_sum = lambda x: x.rolling(window=window_num, min_periods=window_min).sum()

    return f_min, f_max, f_mean, f_std, f_sum

def lag_player_values(df, rel_num_cols):
    
    df = df.sort_values(['GAME_DATE_EST'])

    df = (
        df.assign(**{
        f'player_{col}_lagged': df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
        .reset_index(drop=True)
    )

    df.drop(rel_num_cols, axis=1, inplace=True)

    return df


def create_lagged_player_stats(df, rel_num_cols):
    
    df = df.sort_values(['GAME_DATE_EST'])

    df = (
        df.assign(**{
        f'player_{col}_lagged': df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
        .reset_index(drop=True)
    )

    df.drop(rel_num_cols, axis=1, inplace=True)

    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    for col in df.columns[df.columns.str.endswith('_lagged')]:
        print(col)
        for i in range(len(function_list)):
            df[(col + '_%s' % function_name[i])] = df.sort_values(['GAME_DATE_EST']).groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])


    return df

def create_lagged_team_stats(df, rel_num_cols):
    
    df = df.sort_values(['GAME_DATE_EST'])

    df = (
        df.assign(**{
        f'team_{col}_lagged': df.groupby(['TEAM_ID', 'SEASON_ID'], group_keys=False)[col].shift(1)
        for col in rel_num_cols})
        .reset_index(drop=True)
    )

    df.drop(rel_num_cols, axis=1, inplace=True)

    function_list = [f_min, f_max, f_mean, f_std, f_sum]
    function_name = ['min', 'max', 'mean', 'std', 'sum']

    for col in df.columns[df.columns.str.endswith('_lagged')]:
        print(col)
        for i in range(len(function_list)):
            df[(col + '_%s' % function_name[i])] = df.sort_values(['GAME_DATE_EST']).groupby(['TEAM_ID', 'SEASON_ID'], group_keys=False)[col].apply(function_list[i])
            print(function_name[i])

    return df


# IS THERE TEAM RANKING SOMEWHERE?

# TEAM ROLLING RANK -----------------------------------------------------

# PLAYER ROLLING RANK ----------------------------------------------------
f_min, f_max, f_mean, f_std, f_sum = create_aggregate_rolling_functions()
boxscore_complete_player_processed_processed = create_lagged_player_stats(boxscore_complete_player_processed, rel_box_base_num_cols + rel_box_adv_num_cols)
boxscore_complete_player_processed_processed = lag_player_values(boxscore_complete_player_processed_processed, add_num_cols_player)

boxscore_complete_player_processed_processed = boxscore_complete_player_processed_processed.merge(boxscore_complete_player_processed_processed)
boxscore_complete_player_processed_processed['TEAM_ID'] = boxscore_complete_player_processed_processed['TEAM_ID'].astype(str)

## join with rolling player rank ----------------------
boxscore_complete_player_processed_processed = (
    boxscore_complete_player_processed_processed
    .merge(
    player_season_ranking_calendar_df[['calendar_date', 'PLAYER_ID', 'fantasy_points_rank_overall', 'fantasy_points_rank_position']], 
    left_on=['GAME_DATE_EST', 'PLAYER_ID'],
    right_on=['calendar_date', 'PLAYER_ID'],
    suffixes = [None, '_calendar'],
    how='left'
    )
)






combined_df_reg_base_team_processed = create_lagged_team_stats(combined_df_reg_base_team, rel_box_base_num_cols + rel_box_adv_num_cols + ['fantasy_points_team'])
combined_df_reg_base_team_processed.drop(['game_type', 'SEASON', 'GAME_DATE_EST', 'SEASON_ID'], axis=1, inplace=True)

team_lagged_cols = [col for col in combined_df_reg_base_team_processed.columns if 'lagged' in col]
team_oppossing_stats = combined_df_reg_base_team_processed[rel_box_base_id_cols_team + team_lagged_cols]


combined_df_reg_player_team_processed = boxscore_complete_player_processed_processed.merge(combined_df_reg_base_team_processed, how='left', on=['GAME_ID', 'TEAM_ID'])

combined_df_reg_player_team_processed = combined_df_reg_player_team_processed.merge(team_oppossing_stats, how='left' ,on='GAME_ID', suffixes=[None, '_opposing'])
combined_df_reg_player_team_processed = combined_df_reg_player_team_processed[combined_df_reg_player_team_processed['TEAM_ID'] != combined_df_reg_player_team_processed['TEAM_ID_opposing']]



# TRANSFORMERS ------------------------------------------------------------------

## custom date transformer  ----------------------------------------------------
date_feats = ['dayofweek', 'dayofyear', 'is_leap_year', 'quarter', 'weekofyear', 'year', 'season', 'week']

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
          return self

    def transform(self, x):

        x['GAME_DATE_EST'] = pd.to_datetime(x['GAME_DATE_EST'])

        dayofweek = x.GAME_DATE_EST.dt.dayofweek
        dayofyear= x.GAME_DATE_EST.dt.dayofyear
        is_leap_year =  x.GAME_DATE_EST.dt.is_leap_year
        quarter =  x.GAME_DATE_EST.dt.quarter
        weekofyear = x.GAME_DATE_EST.dt.weekofyear
        year = x.GAME_DATE_EST.dt.year

        df_dt = pd.concat([dayofweek, dayofyear,  is_leap_year, quarter, weekofyear, year], axis=1)

        return df_dt

date_pipeline = Pipeline(steps=[
    ('date', DateTransformer())
])

## numeric transformer --------------------------------------------

# for base model the base columns are all numeric columns

num_cols = [col for col in combined_df_reg_player_team_processed.columns if 'lagged' in col]

# num_cols.append('games_played')
num_cols.append('fantasy_points_rank_overall')
num_cols.append('fantasy_points_rank_position')


num_pipeline = Pipeline(steps=[
    ('scale', StandardScaler())
])

## cat transformer -----------------------------------------------

cat_cols_high_card = ['PLAYER_ID', 'TEAM_ID']

cat_pipeline_high_card = Pipeline(steps=[
    ('encoder', TargetEncoder(smoothing=2))
])


cat_cols_low_card = ['is_starter', 'home_away']

cat_pipeline_low_card = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map, cols=['compass'])


# SET UP VALIDATION METHODS -----------------------------------------------
       
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2

combined_df_reg_player_team_processed['fantasy_points'].value_counts()
combined_df_reg_player_team_processed[combined_df_reg_player_team_processed['fantasy_points'] < 0]['fantasy_points'].value_counts()

# TRAIN MODEL -----------------------------------------------------------
## mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns/ 
remote_server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(remote_server_uri)

exp_name = 'nba_fantasy_regression'
#mlflow.create_experiment(exp_name)

mlflow.set_experiment(exp_name)

with mlflow.start_run() as run:
    
    full_data = combined_df_reg_player_team_processed[['SEASON_ID', 'GAME_DATE_EST','fantasy_points'] + num_cols + cat_cols_high_card + cat_cols_low_card]
    full_data.dropna(axis=0, inplace=True)

    train = full_data[full_data['SEASON_ID'] < 2019]
    test = full_data[full_data['SEASON_ID'] >= 2019]

    X_train = train.drop(['fantasy_points'], axis=1)
    y_train = train['fantasy_points'] + 5

    n_estimators = 5
    max_features = 0.3
    min_samples_leaf = 0.05
    rf = RandomForestRegressor(n_estimators=n_estimators, 
                               #max_depth=max_depth, 
                               max_features=max_features,
                               min_samples_leaf=min_samples_leaf,
                               n_jobs=-1, 
                               random_state=0)

    col_trans_pipeline = ColumnTransformer(
        transformers=[
            ('date', date_pipeline, ['GAME_DATE_EST']),
            ('numeric', num_pipeline, num_cols),
            ('cat_high', cat_pipeline_high_card, cat_cols_high_card),
            ('cat_low', cat_pipeline_low_card, cat_cols_low_card)
        ]
    )

    lr = LinearRegression()
    glm = linear_model.GammaRegressor()

    rf_pipeline = Pipeline(steps=[
        ('preprocess', col_trans_pipeline),
        ('lr', glm)
    ])
    

    tscv = TimeSeriesSplit(n_splits=5)
    cross_val_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=tscv)
    cross_val_score_mean_sqaured_error = cross_val_score(rf_pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    
    rf_pipeline.fit(X_train, y_train)
    y_train_pred = rf_pipeline.predict(X_train) 
    (rmse, mae, r2) = eval_metrics(y_train, y_train_pred)
    
    mlflow.set_tag('mlflow.runName', 'glm_add_5') # set tag with run name so we can search for it later

    #params_dict = {"n_estimators": n_estimators, "n_predictors": rf.n_features_in_,  'max_features': max_features,
              #     'min_samples_leaf': min_samples_leaf}
    
   # mlflow.log_params(params_dict)

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('cross_val_score_avg', cross_val_scores.mean())
    mlflow.log_metric('cross_val_score_rmse', np.mean(np.sqrt(np.abs(cross_val_score_mean_sqaured_error))))
    



mlflow.sklearn.save_model(rf_pipeline, path='../base_adv_model/model_v2_w_rolling_window_20')
