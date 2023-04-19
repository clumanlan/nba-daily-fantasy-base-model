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
- **0-get-initial-data:** Pulls boxscore data (including advanced stats) at the game, team and player level from [nba_api](https://github.com/swar/nba_api)

## Streamlit Dash
