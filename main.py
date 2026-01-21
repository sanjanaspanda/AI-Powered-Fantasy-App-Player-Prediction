# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# import joblib

# app = Flask(__name__)

# # Load the trained model and scalers
# model = load_model('./trained_model.h5')
# scaler_X = joblib.load('./scaler_X.joblib')
# scaler_y = joblib.load('./scaler_y.joblib')

# # Load the aggregated player stats (you'll need to save this during preprocessing)
# player_stats = pd.read_csv('aggregated_player_stats.csv')

# def calculate_fantasy_points(row, match_type='ODI'):
#     points = 0
#     points += 4  # Appearance points
    
#     # Batting points
#     points += row['Runs Scored']
#     if row['Runs Scored'] >= 50:
#         points += 4 if match_type in ['ODI', 'Test'] else 8
#     if row['Runs Scored'] >= 100:
#         points += 8
    
#     # Bowling points
#     points += row['Wickets Taken'] * 25 if match_type != 'Test' else row['Wickets Taken'] * 16
#     if row['Wickets Taken'] >= 4:
#         points += 4
#     if row['Wickets Taken'] >= 5:
#         points += 8
    
#     # Economy rate points (assuming ODI)
#     if row['Balls Bowled'] >= 12:  # Minimum 2 overs
#         economy_rate = (row['Runs Conceded'] / (row['Balls Bowled'] / 6)) * 6
#         if economy_rate < 2.5:
#             points += 6
#         elif economy_rate < 3.5:
#             points += 4
#         elif economy_rate < 4.5:
#             points += 2
#         elif economy_rate > 7:
#             points -= 2
#         elif economy_rate > 8:
#             points -= 4
#         elif economy_rate > 9:
#             points -= 6
    
#     return points

# def predict_player_performance(player_name):
#     player_data = player_stats[player_stats['Player Name'] == player_name].iloc[0]
#     features = ['Batting Average', 'Bowling Average', 'Strike Rate', 'Economy Rate', 'Player Experience']
#     X = player_data[features].values.reshape(1, -1)
#     X_scaled = scaler_X.transform(X)
    
#     y_pred_scaled = model.predict(X_scaled)
#     y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
#     return {
#         'Runs Scored': y_pred[0][0],
#         'Wickets Taken': y_pred[0][1],
#         'Balls Bowled': y_pred[0][2],
#         'Runs Conceded': y_pred[0][3]
#     }

# @app.route('/predict-top-11', methods=['POST'])
# def predict_top_11():
#     try:
#         data = request.get_json()
#         team1 = data['team1']
#         team2 = data['team2']
#         match_type = data.get('match_type', 'ODI')
#         print(player_stats)
#         all_players = player_stats[(player_stats['Team'] == team1) | (player_stats['Team'] == team2)]
#         print(all_players)
#         predictions = []
#         for _, player in all_players.iterrows():
#             pred = predict_player_performance(player['Player Name'])
#             pred['Player Name'] = player['Player Name']
#             pred['Team'] = player['Team']
#             pred['Player Role'] = player['Player Role']  # Assuming you have this in your player_stats
#             pred['Predicted Fantasy Points'] = calculate_fantasy_points(pred, match_type)
#             predictions.append(pred)
        
#         predictions_df = pd.DataFrame(predictions)
#         top_11 = predictions_df.nlargest(11, 'Predicted Fantasy Points')
        
#         response = {
#             "top_players": top_11.to_dict(orient='records'),
#             "role_distribution": top_11['Player Role'].value_counts().to_dict()
#         }
        
#         return jsonify(response), 200
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the trained model and scalers at startup
model = load_model('./trained_model.h5')
scaler_X = joblib.load('./scaler_X.joblib')
scaler_y = joblib.load('./scaler_y.joblib')

# Load the aggregated player stats
player_stats = pd.read_csv('aggregated_player_stats.csv')

# Fill missing values (NaNs) with default values to avoid NaNs during prediction
player_stats.fillna({
    'Batting Average': 0,
    'Bowling Average': 0,
    'Strike Rate': 0,
    'Economy Rate': 0,
    'Player Experience': 0,
    'Runs Scored': 0,
    'Wickets Taken': 0,
    'Balls Bowled': 0,
    'Runs Conceded': 0
}, inplace=True)

def calculate_fantasy_points(row, match_type='ODI'):
    points = 0
    points += 4  # Appearance points
    
    # Batting points
    points += row['Runs Scored']
    if row['Runs Scored'] >= 50:
        points += 4 if match_type in ['ODI', 'Test'] else 8
    if row['Runs Scored'] >= 100:
        points += 8
    
    # Bowling points
    points += row['Wickets Taken'] * 25 if match_type != 'Test' else row['Wickets Taken'] * 16
    if row['Wickets Taken'] >= 4:
        points += 4
    if row['Wickets Taken'] >= 5:
        points += 8
    
    # Economy rate points (assuming ODI)
    if row['Balls Bowled'] >= 12:  # Minimum 2 overs
        economy_rate = (row['Runs Conceded'] / (row['Balls Bowled'] / 6)) * 6
        if economy_rate < 2.5:
            points += 6
        elif economy_rate < 3.5:
            points += 4
        elif economy_rate < 4.5:
            points += 2
        elif economy_rate > 7:
            points -= 2
        elif economy_rate > 8:
            points -= 4
        elif economy_rate > 9:
            points -= 6
    
    return points

def predict_player_performance(player_df):
    features = ['Batting Average', 'Bowling Average', 'Strike Rate', 'Economy Rate', 'Player Experience']
    X = player_df[features].values
    X_scaled = scaler_X.transform(X)
    
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return pd.DataFrame(y_pred, columns=['Runs Scored', 'Wickets Taken', 'Balls Bowled', 'Runs Conceded'])

@app.route('/predict-top-11', methods=['POST'])
def predict_top_11():
    try:
        data = request.get_json()
        team1 = data['team1']
        team2 = data['team2']
        match_type = data.get('match_type', 'ODI')
        
        # Filter players from both teams
        all_players = player_stats[(player_stats['Team'] == team1) | (player_stats['Team'] == team2)].reset_index(drop=True)
        
        # Perform predictions for all players in one batch
        player_predictions = predict_player_performance(all_players)
        
        # Reset index to avoid misalignment during concatenation
        player_predictions.reset_index(drop=True, inplace=True)
        all_players.reset_index(drop=True, inplace=True)
        
        # Combine predictions with player details
        predictions = pd.concat([all_players[['Player Name', 'Team', 'Player Role']], player_predictions], axis=1)
        
        # Calculate fantasy points
        predictions['Predicted Fantasy Points'] = predictions.apply(calculate_fantasy_points, axis=1, match_type=match_type)
        
        # Get top 11 players based on fantasy points
        top_11 = predictions.nlargest(11, 'Predicted Fantasy Points')
        
        response = {
            "top_players": top_11.to_dict(orient='records'),
            "role_distribution": top_11['Player Role'].value_counts().to_dict()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
