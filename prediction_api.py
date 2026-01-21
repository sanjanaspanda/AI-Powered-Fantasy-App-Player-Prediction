# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, LSTM, GRU, Input, Concatenate, Reshape

# data = pd.read_csv('data.csv')

# label_enc = LabelEncoder()
# data['Player Role'] = label_enc.fit_transform(data['Player Role'])
# data['Team'] = label_enc.fit_transform(data['Team'])

# features = ['Batting Average', 'Bowling Average', 'Strike Rate', 'Economy Rate', 
#             'Centuries Scored', 'Half Centuries Scored', 'Ducks Scored', 
#             'Wickets Taken Last Match', 'Runs Scored Last Match', 
#             'Player Age', 'Player Experience', 'Player Role', 'Team']

# X = data[features]
# y = data[['Runs Scored', 'Wickets Taken', 'Balls Faced', 'Balls Bowled', 
#           'Overs Bowled', 'Maidens Bowled', 'Runs Conceded']]

# for col in y.columns:
#     y[col] = y[col] / data['Player Experience']

# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# def create_model(input_shape):
#     input_layer = Input(shape=(input_shape,))
    
#     reshaped_input = Reshape((input_shape, 1))(input_layer)
    
#     lstm = LSTM(64, return_sequences=True)(reshaped_input)
#     lstm = LSTM(32)(lstm)
    
#     gru = GRU(64, return_sequences=True)(reshaped_input)
#     gru = GRU(32)(gru)
    
#     concat = Concatenate()([lstm, gru])
    
#     dense1 = Dense(64, activation='relu')(concat)
#     dense2 = Dense(32, activation='relu')(dense1)
    
#     output = Dense(7)(dense2)  
    
#     model = Model(inputs=input_layer, outputs=output)
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
#     return model

# model = create_model(X_train.shape[1])
# model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test))

# from flask import Flask, request, jsonify

# app = Flask(__name__)

# def calculate_fantasy_points(row, match_type):
#     points = 0
#     points += 4  
#     points += row['Runs Scored']
#     points += row['Wickets Taken'] * 25 if match_type != 'Test' else row['Wickets Taken'] * 16
    
#     if row['Runs Scored'] == 0 and row['Balls Faced'] > 0:
#         points -= 2 if match_type in ['T20', 'T10'] else 3 if match_type == 'ODI' else 4
    
#     if row['Runs Scored'] >= 50:
#         points += 8 if match_type == 'T20' else 4 if match_type in ['ODI', 'Test'] else 16
#     if row['Runs Scored'] >= 100:
#         points += 16 if match_type == 'T20' else 8
    
#     if row['Maidens Bowled'] > 0:
#         points += row['Maidens Bowled'] * (12 if match_type == 'T20' else 4 if match_type == 'ODI' else 16)
#     if row['Wickets Taken'] >= 4:
#         points += 8 if match_type == 'T20' else 4
#     if row['Wickets Taken'] >= 5:
#         points += 16 if match_type == 'T20' else 8
    
#     if row['Overs Bowled'] >= 2:
#         economy_rate = row['Runs Conceded'] / row['Overs Bowled']
#         if economy_rate < 6:
#             points += 4
#         elif economy_rate > 9:
#             points -= 2
    
#     return points

# def get_top_11_players_for_teams(data, team_1, team_2, match_type='ODI'):
#     team_data = data[data['Team'].isin([team_1, team_2])]
#     X_team = team_data[features]
#     X_team_scaled = scaler_X.transform(X_team)
    
#     predicted_stats_scaled = model.predict(X_team_scaled)
#     predicted_stats = scaler_y.inverse_transform(predicted_stats_scaled)
    
#     predicted_df = pd.DataFrame(predicted_stats, columns=y.columns)
#     predicted_df['Player Name'] = team_data['Player Name'].values
#     predicted_df['Team'] = team_data['Team'].values
#     predicted_df['Player Role'] = team_data['Player Role'].values
    
#     for col in y.columns:
#         if col in ['Runs Scored', 'Balls Faced', 'Balls Bowled', 'Runs Conceded']:
#             predicted_df[col] = predicted_df[col].round().astype(int)
#         elif col in ['Wickets Taken', 'Maidens Bowled']:
#             predicted_df[col] = predicted_df[col].round(1)
#         elif col == 'Overs Bowled':
#             predicted_df[col] = predicted_df[col].round(1)
    
#     predicted_df['Predicted Fantasy Points'] = predicted_df.apply(lambda row: calculate_fantasy_points(row, match_type), axis=1)
    
#     return select_top_11_with_roles(predicted_df)

# def select_top_11_with_roles(df):
#     sorted_players = df.sort_values(by='Predicted Fantasy Points', ascending=False)
    
#     team = []
#     roles_covered = set()
    
#     for _, player in sorted_players.iterrows():
#         if len(team) >= 11 and len(roles_covered) == len(label_enc.classes_):
#             break
#         if player['Player Role'] not in roles_covered or len(team) < 11:
#             team.append(player)
#             roles_covered.add(player['Player Role'])
    
#     while len(team) < 11:
#         for _, player in sorted_players.iterrows():
#             if player not in team:
#                 team.append(player)
#                 break
    
#     return pd.DataFrame(team)


# @app.route('/predict-top-11', methods=['POST'])
# def predict_top_11():
#     try:
#         data1 = request.get_json()

#         team1 = data1['team1']
#         team2 = data1['team2']
#         match_type = data1.get('match_type', 'ODI')

#         team_1_encoded = label_enc.transform([team1])[0]
#         team_2_encoded = label_enc.transform([team2])[0]

#         top_players = get_top_11_players_for_teams(data, team_1_encoded, team_2_encoded, match_type)
#         print('top players', top_players)
#         response = {
#             "top_players": top_players.to_dict(orient='records'),
#             "role_distribution": top_players['Player Role'].value_counts().to_dict()
#         }
#         return jsonify(response), 200
    
#     except Exception as e:
#         print(e)
#         return jsonify({"error": str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, LSTM, GRU, Input, Concatenate, Reshape
# from flask import Flask, request, jsonify

# # Load and preprocess data
# data = pd.read_csv('data.csv')

# # Check columns in the DataFrame
# print("Data columns:", data.columns)

# # Encode categorical columns
# label_enc = LabelEncoder()
# data['Player Role'] = label_enc.fit_transform(data['Player Role'])
# data['Team'] = label_enc.fit_transform(data['Team'])

# features = ['Batting Average', 'Bowling Average', 'Strike Rate', 'Economy Rate', 
#             'Centuries Scored', 'Half Centuries Scored', 'Ducks Scored', 
#             'Wickets Taken Last Match', 'Runs Scored Last Match', 
#             'Player Age', 'Player Experience', 'Player Role', 'Team']

# X = data[features]
# y = data[['Runs Scored', 'Wickets Taken', 'Balls Faced', 'Balls Bowled', 
#           'Overs Bowled', 'Maidens Bowled', 'Runs Conceded']]

# # Normalize y values
# for col in y.columns:
#     y[col] = y[col] / data['Player Experience']

# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# # Create and train model
# def create_model(input_shape):
#     input_layer = Input(shape=(input_shape,))
#     reshaped_input = Reshape((input_shape, 1))(input_layer)
    
#     lstm = LSTM(64, return_sequences=True)(reshaped_input)
#     lstm = LSTM(32)(lstm)
    
#     gru = GRU(64, return_sequences=True)(reshaped_input)
#     gru = GRU(32)(gru)
    
#     concat = Concatenate()([lstm, gru])
#     dense1 = Dense(64, activation='relu')(concat)
#     dense2 = Dense(32, activation='relu')(dense1)
#     output = Dense(7)(dense2)  
    
#     model = Model(inputs=input_layer, outputs=output)
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#     return model

# # Train the model
# model = create_model(X_train.shape[1])
# model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test))

# # Calculate fantasy points
# def calculate_fantasy_points(row, match_type):
#     points = 0
#     points += 4
#     points += row['Runs Scored']
#     points += row['Wickets Taken'] * 25 if match_type != 'Test' else row['Wickets Taken'] * 16
    
#     if row['Runs Scored'] == 0 and row['Balls Faced'] > 0:
#         points -= 2 if match_type in ['T20', 'T10'] else 3 if match_type == 'ODI' else 4
#     if row['Runs Scored'] >= 50:
#         points += 8 if match_type == 'T20' else 4 if match_type in ['ODI', 'Test'] else 16
#     if row['Runs Scored'] >= 100:
#         points += 16 if match_type == 'T20' else 8
#     if row['Maidens Bowled'] > 0:
#         points += row['Maidens Bowled'] * (12 if match_type == 'T20' else 4 if match_type == 'ODI' else 16)
#     if row['Wickets Taken'] >= 4:
#         points += 8 if match_type == 'T20' else 4
#     if row['Wickets Taken'] >= 5:
#         points += 16 if match_type == 'T20' else 8
#     if row['Overs Bowled'] >= 2:
#         economy_rate = row['Runs Conceded'] / row['Overs Bowled']
#         if economy_rate < 6:
#             points += 4
#         elif economy_rate > 9:
#             points -= 2
#     return points

# # Get top 11 players
# def get_top_11_players_for_teams(data, team_1, team_2, match_type='ODI'):
#     print("Encoded teams:", team_1, team_2)
#     print("Data sample:", data.head())
    
#     team_data = data[data['Team'].isin([team_1, team_2])]
#     print("Filtered team data:", team_data.head())
    
#     X_team = team_data[features]
#     X_team_scaled = scaler_X.transform(X_team)
    
#     predicted_stats_scaled = model.predict(X_team_scaled)
#     predicted_stats = scaler_y.inverse_transform(predicted_stats_scaled)
    
#     predicted_df = pd.DataFrame(predicted_stats, columns=y.columns)
#     predicted_df['Player Name'] = team_data['Player Name'].values
#     predicted_df['Team'] = team_data['Team'].values
#     predicted_df['Player Role'] = team_data['Player Role'].values
    
#     for col in y.columns:
#         if col in ['Runs Scored', 'Balls Faced', 'Balls Bowled', 'Runs Conceded']:
#             predicted_df[col] = predicted_df[col].round().astype(int)
#         elif col in ['Wickets Taken', 'Maidens Bowled']:
#             predicted_df[col] = predicted_df[col].round(1)
#         elif col == 'Overs Bowled':
#             predicted_df[col] = predicted_df[col].round(1)
    
#     predicted_df['Predicted Fantasy Points'] = predicted_df.apply(lambda row: calculate_fantasy_points(row, match_type), axis=1)
#     return select_top_11_with_roles(predicted_df)

# # Select top 11 players with roles
# def select_top_11_with_roles(df):
#     sorted_players = df.sort_values(by='Predicted Fantasy Points', ascending=False)
#     team = []
#     roles_covered = set()
    
#     for _, player in sorted_players.iterrows():
#         if len(team) >= 11 and len(roles_covered) == len(label_enc.classes_):
#             break
#         if player['Player Role'] not in roles_covered or len(team) < 11:
#             team.append(player)
#             roles_covered.add(player['Player Role'])
    
#     while len(team) < 11:
#         for _, player in sorted_players.iterrows():
#             if player not in team:
#                 team.append(player)
#                 break
#     return pd.DataFrame(team)

# # Flask app
# app = Flask(__name__)

# @app.route('/predict-top-11', methods=['POST'])
# def predict_top_11():
#     try:
#         data1 = request.get_json()
#         team1 = data1['team1']
#         team2 = data1['team2']
#         match_type = data1.get('match_type', 'ODI')
        
#         team_1_encoded = label_enc.transform([team1])[0]
#         team_2_encoded = label_enc.transform([team2])[0]
        
#         top_players = get_top_11_players_for_teams(data, team_1_encoded, team_2_encoded, match_type)
        
#         response = {
#             "top_players": top_players.to_dict(orient='records'),
#             "role_distribution": top_players['Player Role'].value_counts().to_dict()
#         }
#         return jsonify(response), 200
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from tensorflow.keras.models import load_model
# import joblib
# from flask import Flask, request, jsonify

# # Load the model and scalers
# model = load_model('trained_model.h5')
# scaler_X = joblib.load('scaler_X.pkl')
# scaler_y = joblib.load('scaler_y.pkl')
# label_enc = joblib.load('label_enc.pkl')

# # Define the features and target columns
# features = ['Batting Average', 'Bowling Average', 'Strike Rate', 'Economy Rate', 
#             'Centuries Scored', 'Half Centuries Scored', 'Ducks Scored', 
#             'Wickets Taken Last Match', 'Runs Scored Last Match', 
#             'Player Age', 'Player Experience', 'Player Role', 'Team']

# y_columns = ['Runs Scored', 'Wickets Taken', 'Balls Faced', 'Balls Bowled', 
#              'Overs Bowled', 'Maidens Bowled', 'Runs Conceded']

# # Calculate fantasy points
# def calculate_fantasy_points(row, match_type):
#     points = 0
#     points += 4
#     points += row['Runs Scored']
#     points += row['Wickets Taken'] * 25 if match_type != 'Test' else row['Wickets Taken'] * 16
    
#     if row['Runs Scored'] == 0 and row['Balls Faced'] > 0:
#         points -= 2 if match_type in ['T20', 'T10'] else 3 if match_type == 'ODI' else 4
#     if row['Runs Scored'] >= 50:
#         points += 8 if match_type == 'T20' else 4 if match_type in ['ODI', 'Test'] else 16
#     if row['Runs Scored'] >= 100:
#         points += 16 if match_type == 'T20' else 8
#     if row['Maidens Bowled'] > 0:
#         points += row['Maidens Bowled'] * (12 if match_type == 'T20' else 4 if match_type == 'ODI' else 16)
#     if row['Wickets Taken'] >= 4:
#         points += 8 if match_type == 'T20' else 4
#     if row['Wickets Taken'] >= 5:
#         points += 16 if match_type == 'T20' else 8
#     if row['Overs Bowled'] >= 2:
#         economy_rate = row['Runs Conceded'] / row['Overs Bowled']
#         if economy_rate < 6:
#             points += 4
#         elif economy_rate > 9:
#             points -= 2
#     return points

# # Get top 11 players
# def get_top_11_players_for_teams(data, team_1, team_2, match_type='ODI'):
#     team_data = data[data['Team'].isin([team_1, team_2])]
#     X_team = team_data[features]
#     X_team_scaled = scaler_X.transform(X_team)
    
#     predicted_stats_scaled = model.predict(X_team_scaled)
#     predicted_stats = scaler_y.inverse_transform(predicted_stats_scaled)
    
#     predicted_df = pd.DataFrame(predicted_stats, columns=y_columns)
#     predicted_df['Player Name'] = team_data['Player Name'].values
#     predicted_df['Team'] = team_data['Team'].values
#     predicted_df['Player Role'] = team_data['Player Role'].values
    
#     for col in y_columns:
#         if col in ['Runs Scored', 'Balls Faced', 'Balls Bowled', 'Runs Conceded']:
#             predicted_df[col] = predicted_df[col].round().astype(int)
#         elif col in ['Wickets Taken', 'Maidens Bowled']:
#             predicted_df[col] = predicted_df[col].round(1)
#         elif col == 'Overs Bowled':
#             predicted_df[col] = predicted_df[col].round(1)
    
#     predicted_df['Predicted Fantasy Points'] = predicted_df.apply(lambda row: calculate_fantasy_points(row, match_type), axis=1)
#     return select_top_11_with_roles(predicted_df)

# # Select top 11 players with roles
# def select_top_11_with_roles(df):
#     sorted_players = df.sort_values(by='Predicted Fantasy Points', ascending=False)
#     team = []
#     roles_covered = set()
    
#     for _, player in sorted_players.iterrows():
#         if len(team) >= 11 and len(roles_covered) == len(label_enc.classes_):
#             break
#         if player['Player Role'] not in roles_covered or len(team) < 11:
#             team.append(player)
#             roles_covered.add(player['Player Role'])
    
#     while len(team) < 11:
#         for _, player in sorted_players.iterrows():
#             if player not in team:
#                 team.append(player)
#                 break
#     return pd.DataFrame(team)

# # Flask app
# app = Flask(__name__)

# @app.route('/predict-top-11', methods=['POST'])
# def predict_top_11():
#     try:
#         data = request.get_json()
#         team1 = data['team1']
#         team2 = data['team2']
#         match_type = data.get('match_type', 'ODI')
        
#         team_1_encoded = label_enc.transform([team1])[0]
#         team_2_encoded = label_enc.transform([team2])[0]
        
#         top_players = get_top_11_players_for_teams(data, team_1_encoded, team_2_encoded, match_type)
        
#         response = {
#             "top_players": top_players.to_dict(orient='records'),
#             "role_distribution": top_players['Player Role'].value_counts().to_dict()
#         }
#         return jsonify(response), 200
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Input, Concatenate, Reshape
import joblib
import os
from flask import Flask, request, jsonify

data = pd.read_csv('./data.csv')
features = ['Batting Average', 'Bowling Average', 'Strike Rate', 'Economy Rate', 
            'Centuries Scored', 'Half Centuries Scored', 'Ducks Scored', 
            'Wickets Taken Last Match', 'Runs Scored Last Match', 
            'Player Age', 'Player Experience', 'Player Role', 'Team']
y = data[['Runs Scored', 'Wickets Taken', 'Balls Faced', 'Balls Bowled', 
          'Overs Bowled', 'Maidens Bowled', 'Runs Conceded', 'Catch taken', 'Caught & Bowled', 'Stumping/Run Out (direct)', 'Run Out (Thrower/Catcher)',]]
# If the model exists, load it
model = load_model('trained_model.h5')

# Load the scalers
scaler_X = joblib.load('scaler_x.joblib')
scaler_y = joblib.load('scaler_y.joblib')

# Load the LabelEncoder
label_enc = joblib.load('label_enc.joblib')

# Fantasy points calculation function
# def calculate_fantasy_points(row, match_type):
#     points = 0
#     points += 4  # Appearance points
#     points += row['Runs Scored']
#     points += row['Wickets Taken'] * 25 if match_type != 'Test' else row['Wickets Taken'] * 16
    
#     # Duck penalty
#     if row['Runs Scored'] == 0 and row['Balls Faced'] > 0:
#         points -= 2 if match_type in ['T20', 'T10'] else 3 if match_type == 'ODI' else 4
    
#     # Batting bonuses
#     if row['Runs Scored'] >= 50:
#         points += 8 if match_type == 'T20' else 4 if match_type in ['ODI', 'Test'] else 16
#     if row['Runs Scored'] >= 100:
#         points += 16 if match_type == 'T20' else 8
    
#     # Bowling bonuses
#     if row['Maidens Bowled'] > 0:
#         points += row['Maidens Bowled'] * (12 if match_type == 'T20' else 4 if match_type == 'ODI' else 16)
#     if row['Wickets Taken'] >= 4:
#         points += 8 if match_type == 'T20' else 4
#     if row['Wickets Taken'] >= 5:
#         points += 16 if match_type == 'T20' else 8
    
#     # Economy rate bonuses/penalties
#     if row['Overs Bowled'] >= 2:
#         economy_rate = row['Runs Conceded'] / row['Overs Bowled']
#         if economy_rate < 6:
#             points += 4
#         elif economy_rate > 9:
#             points -= 2
    
#     return points
def calculate_fantasy_points(row, match_type):
    points = 0
    points += 4

    points += row['Runs Scored']

    if match_type == 'T20' or match_type == 'T10':
        points += row['Wickets Taken'] * 25
    elif match_type == 'ODI':
        points += row['Wickets Taken'] * 25
    elif match_type == 'Test':
        points += row['Wickets Taken'] * 16

    points += row['Catch taken'] * 8

    points += row['Caught & Bowled'] * 33

    points += row['Stumping/Run Out (direct)'] * 12

    points += row['Run Out (Thrower/Catcher)'] * 6

    if row['Runs Scored'] == 0 and row['Balls Faced'] > 0:
        if match_type == 'T20' or match_type == 'T10':
            points -= 2
        elif match_type == 'ODI':
            points -= 3
        elif match_type == 'Test':
            points -= 4

    if row['Runs Scored'] >= 50:
        if match_type == 'T20':
            points += 8
        elif match_type in ['ODI', 'Test']:
            points += 4
    if row['Runs Scored'] >= 100:
        if match_type == 'T20':
            points += 16
        elif match_type in ['ODI', 'Test']:
            points += 8

    if row['Maidens Bowled'] > 0:
        if match_type == 'T20':
            points += row['Maidens Bowled'] * 12
        elif match_type == 'ODI':
            points += row['Maidens Bowled'] * 4
        elif match_type == 'Test':
            points += row['Maidens Bowled'] * 0 

    if row['Wickets Taken'] >= 4:
        if match_type == 'T20':
            points += 8
        elif match_type == 'ODI':
            points += 4
    if row['Wickets Taken'] >= 5:
        if match_type == 'T20':
            points += 16
        elif match_type == 'ODI':
            points += 8

    if row['Overs Bowled'] >= 2:
        economy_rate = row['Runs Conceded'] / row['Overs Bowled']
        if match_type == 'T20':
            if economy_rate < 5:
                points += 6
            elif 5 <= economy_rate < 6:
                points += 4
            elif 6 <= economy_rate < 7:
                points += 2
            elif 10 <= economy_rate < 11:
                points -= 2
            elif 11 <= economy_rate < 12:
                points -= 4
            elif economy_rate > 12:
                points -= 6
        elif match_type == 'ODI':
            if economy_rate < 2.5:
                points += 6
            elif 2.5 <= economy_rate < 3.5:
                points += 4
            elif 3.5 <= economy_rate < 4.5:
                points += 2
            elif 7 <= economy_rate < 8:
                points -= 2
            elif 8 <= economy_rate < 9:
                points -= 4
            elif economy_rate > 9:
                points -= 6
        elif match_type == 'Test':
            if economy_rate < 2.5:
                points += 6
            elif 2.5 <= economy_rate < 3.5:
                points += 4
            elif 3.5 <= economy_rate < 4.5:
                points += 2
            elif economy_rate > 9:
                points -= 6
        elif match_type == 'T10':
            if economy_rate < 7:
                points += 6
            elif 7 <= economy_rate < 8:
                points += 4
            elif 8 <= economy_rate < 9:
                points += 2
            elif 14 <= economy_rate < 15:
                points -= 2
            elif 15 <= economy_rate < 16:
                points -= 4
            elif economy_rate > 16:
                points -= 6

    return points

def update_playing_11(df, player_ids):
    df['In Playing 11'] = df['id'].apply(lambda x: 1 if x in player_ids else 0)
    return df
# Function to select top 11 players from specific teams
def get_top_11_players_for_teams(data: pd.DataFrame, team_1, team_2, player_ids, match_type='ODI'):
    # Filter data for the specified teams
    team_data = data[data['Team'].isin([team_1, team_2])]
    # Prepare the data for prediction
    team_1_encoded = label_enc.transform([team_1])[0]
    team_2_encoded = label_enc.transform([team_2])[0]
    team_data.replace(team_1, team_1_encoded, inplace=True)
    team_data.replace(team_2, team_2_encoded, inplace=True)
    X_team = team_data[features]    
    X_team_scaled = scaler_X.transform(X_team)
    # Predict stats for the filtered dataset
    predicted_stats_scaled = model.predict(X_team_scaled)
    predicted_stats = scaler_y.inverse_transform(predicted_stats_scaled)
    
    predicted_df = pd.DataFrame(predicted_stats, columns=y.columns)
    predicted_df['Player Name'] = team_data['Player Name'].values
    predicted_df['id'] = team_data['id'].values
    predicted_df['Team'] = team_data['Team'].values
    predicted_df['Player Role'] = team_data['Player Role'].values
    
    for col in y.columns:
        if col in ['Runs Scored', 'Balls Faced', 'Balls Bowled', 'Runs Conceded', "Catch taken", "Caught & Bowled", "Overs Bowled", "Run Out (Thrower/Catcher)", "Stumping/Run Out (direct)", "Wickets Taken"]:
            predicted_df[col] = predicted_df[col].round().astype(int)
        elif col in ['Wickets Taken', 'Maidens Bowled']:
            predicted_df[col] = predicted_df[col].round(1).astype(int)
        elif col == 'Overs Bowled':
            predicted_df[col] = predicted_df[col].round(1).astype(int)
    
    predicted_df['Predicted Fantasy Points'] = predicted_df.apply(lambda row: calculate_fantasy_points(row, match_type), axis=1)
    team_data = update_playing_11(predicted_df, player_ids=player_ids)
    return select_top_11_with_roles(predicted_df)

# Top 11 player selection function with role consideration
def select_top_11_with_roles(df):
    print(df)
    playing_11_df = df[df['In Playing 11'] == 1]
    sorted_players = playing_11_df.sort_values(by='Predicted Fantasy Points', ascending=False)
    
    team = []
    roles_covered = set()
    teams_covered = set()
    
    # Ensure at least one player from each team is selected
    for team_id in sorted_players['Team'].unique():
        team_players = sorted_players[sorted_players['Team'] == team_id]
        if not team_players.empty:
            best_player = team_players.iloc[0]
            team.append(best_player)
            roles_covered.add(best_player['Player Role'])
            teams_covered.add(team_id)
    
    # Fill the remaining spots
    for _, player in sorted_players.iterrows():
        if len(team) >= 11 and len(roles_covered) == len(df['Player Role'].unique()):
            break
        if player['Player Role'] not in roles_covered or len(team) < 11:
            if player['Player Name'] not in [p['Player Name'] for p in team]:
                team.append(player)
                roles_covered.add(player['Player Role'])
    
    # If we still don't have 11 players, add the next best available
    while len(team) < 11:
        for _, player in sorted_players.iterrows():
            if player['Player Name'] not in [p['Player Name'] for p in team]:
                team.append(player)
                break
    
    return pd.DataFrame(team)

# Example usage
# team_1 = 'Sri Lanka'
# team_2 = 'Australia'
# # Filter data based on team names
# top_players = get_top_11_players_for_teams(data, team_1, team_2)

# print("Top 11 Players based on Predicted Fantasy Points:")
# print(top_players[['Player Name', 'Team', 'Predicted Fantasy Points', 'Player Role']])
app = Flask(__name__)

@app.route('/predict-top-11', methods=['POST'])
def predict_top_11():
    try:
        data1 = request.get_json()
        team1 = data1['team1']
        team2 = data1['team2']
        match_type = data1.get('match_type', 'ODI')
        player_ids = data1.get('player_ids')
        
        team_1_encoded = label_enc.transform([team1])[0]
        team_2_encoded = label_enc.transform([team2])[0]
        print(player_ids)
        top_players = get_top_11_players_for_teams(data, team1, team2, player_ids, match_type)
        top_players["Team"].replace(team_1_encoded, team1, inplace=True)
        top_players["Team"].replace(team_2_encoded, team2, inplace=True)
        response = {
            "top_players": top_players.to_dict(orient='records'),
            "role_distribution": top_players['Player Role'].value_counts().to_dict()
        }
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)