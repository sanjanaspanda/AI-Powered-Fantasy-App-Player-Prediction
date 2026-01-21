# Cricket Player Performance Prediction and Fantasy Points Calculation

## Overview

This project is designed to predict the performance of cricket players based on their historical data and calculate their potential fantasy points in different match formats. The ultimate goal is to assist in selecting the top-performing players for a fantasy cricket team.

### Key Objectives:

- Predict player performance metrics using machine learning.
- Calculate fantasy points based on predicted performance.
- Select a top 11-player team based on predicted fantasy points, ensuring all player roles are covered.

## Data Preprocessing

### 1. **Data Loading and Understanding**

The first step in the project is to load the dataset containing player statistics. The dataset includes various features like batting average, bowling average, strike rate, economy rate, and other cricket-related metrics. Additionally, player-specific information like age, experience, role, and team is also included.

### 2. **Label Encoding**

Since the dataset contains categorical variables (e.g., `Player Role`, `Team`), these are converted into numerical format using label encoding. Label encoding assigns a unique integer to each category, allowing the machine learning model to process these features effectively.

### 3. **Feature Selection**

Specific features are chosen to train the model. These include statistical measures (e.g., batting average), recent performance metrics (e.g., runs scored last match), and player-specific attributes (e.g., player role, team). These features are considered because they are likely to have a significant impact on predicting player performance.

### 4. **Target Variable Transformation**

The target variables, which include various performance metrics (e.g., runs scored, wickets taken), are normalized by dividing them by the player's experience. This normalization helps in stabilizing the variance in the data, leading to better model performance.

### 5. **Scaling**

To ensure that all features contribute equally to the model, both the selected features and target variables are standardized. Standardization scales the features and targets to have a mean of 0 and a standard deviation of 1. This step is crucial when dealing with models that are sensitive to the scale of input data, like neural networks.

### 6. **Data Splitting**

The dataset is split into training and test sets to evaluate the model's performance on unseen data. The training set is used to train the model, while the test set is reserved for evaluating how well the model generalizes.

## Model Architecture

### 1. **Model Design**

The model is designed using a hybrid neural network architecture that combines Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) layers. Both LSTM and GRU are types of recurrent neural networks (RNNs) known for their effectiveness in handling sequential data and capturing temporal dependencies.

### 2. **Input Layer and Reshaping**

The input layer accepts the feature set. Since RNN layers expect 3D input, the feature data is reshaped accordingly. This reshaping allows the model to process each feature as a sequence, even though in this context, the sequence length is effectively 1.

### 3. **LSTM and GRU Layers**

- **LSTM Layers:** Capture long-term dependencies in the data. LSTM is particularly effective at learning patterns in time series data, making it suitable for predicting sports performance metrics.
- **GRU Layers:** Similar to LSTM but with fewer parameters, GRU layers capture essential dependencies without overcomplicating the model.

By combining LSTM and GRU layers, the model leverages the strengths of both architectures, capturing complex patterns in the data.

### 4. **Concatenation and Dense Layers**

The outputs from the LSTM and GRU layers are concatenated, creating a combined representation of the input data. This representation is then passed through fully connected (dense) layers that apply non-linear transformations, enabling the model to learn complex relationships between features.

### 5. **Output Layer**

The output layer produces predictions for the target variables. Since multiple outputs are required (e.g., runs scored, wickets taken), the output layer is designed to produce multiple values simultaneously.

## Training the Model

### 1. **Compilation**

The model is compiled using an optimizer (Adam) and a loss function (mean squared error). The optimizer adjusts the model parameters to minimize the loss, while the loss function measures the difference between predicted and actual values.

### 2. **Training Process**

The model is trained for multiple epochs, where each epoch involves the model making predictions on the training data, calculating the loss, and updating the model weights. A small batch size is chosen to ensure that the model updates frequently, which can be beneficial when training on small datasets.

### 3. **Validation**

During training, the model’s performance is evaluated on the validation set (a subset of the test set). This helps monitor overfitting, ensuring that the model performs well on unseen data.

## Prediction and Inverse Transformation

### 1. **Making Predictions**

Once the model is trained, it predicts performance metrics for all players in the dataset. These predictions are made on scaled data, so the outputs need to be transformed back to their original scale.

### 2. **Inverse Scaling**

The predicted values are transformed back to the original scale using the inverse of the standardization process. This step ensures that the predictions are in a meaningful range that can be interpreted and compared to actual values.

### 3. **Post-processing Predictions**

The predicted values are rounded to reflect realistic cricket statistics (e.g., whole numbers for runs scored, wickets taken). Some performance metrics, such as overs bowled, may be rounded to one decimal place to maintain accuracy.

## Fantasy Points Calculation

### 1. **Fantasy Points Logic**

The project includes a function to calculate fantasy points based on the predicted performance metrics. Fantasy points are calculated differently depending on the match type (e.g., T20, ODI, Test).

- **Runs Scored:** Points are awarded based on the number of runs scored, with additional bonuses for scoring 50 or 100 runs.
- **Wickets Taken:** Higher points are awarded for taking wickets, with additional bonuses for taking 4 or 5 wickets.
- **Economy Rate:** Points are adjusted based on the bowler's economy rate, with penalties for expensive bowling.

### 2. **Match Type Consideration**

The fantasy points calculation logic differs based on the match type. For example, wickets taken in T20s are awarded more points than in Tests due to the shorter format and higher impact of each wicket.

## Team Selection

### 1. **Selecting Top 11 Players**

After calculating fantasy points for all players, the top 11 players are selected based on their predicted fantasy points. The selection process ensures that all player roles (e.g., batsman, bowler, all-rounder) are covered.

### 2. **Role-Based Filtering**

The selection process first ensures that each role is represented in the team. After role-based filtering, the remaining slots are filled by the highest-scoring players, regardless of their role.

### 3. **Final Team Composition**

The final team composition is determined by balancing the roles and maximizing the total predicted fantasy points. This approach ensures a well-rounded team capable of scoring high in fantasy leagues.

## Dependencies

### 1. **Python Libraries**

- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computations.
- **Scikit-Learn:** For data preprocessing, including scaling, label encoding, and train-test splitting.
- **TensorFlow/Keras:** For building, training, and evaluating the neural network model.

### 2. **Dataset**

The project relies on a dataset containing historical cricket player statistics. The dataset should include features related to player performance, experience, and categorical information like player role and team.

## Limitations

### 1. **Limited Features**

The current model uses a limited set of features primarily focused on player statistics. While these features are significant, they do not capture the full context of a cricket match or a player's performance. For example, the model does not consider team dynamics, match conditions, or opposition strength, which can have a significant impact on player performance.

### 2. **Lack of Contextual and Environmental Data**

The model does not account for external factors such as weather conditions, pitch type, or match format. These factors can significantly influence player performance, especially in cricket, where conditions vary greatly from one match to another. The absence of these features may limit the accuracy and generalizability of the predictions.

### 3. **No Consideration for Opponent Strength**

The model does not include any information about the strength or ratings of the opposing team or players. The performance of a player can vary greatly depending on the quality of the opposition, and without this data, the predictions might be overly optimistic or pessimistic.

### 4. **Injury Status Not Considered**

The model does not take into account the injury status of players. A player's current fitness level is a critical factor in performance prediction, as injured players may not perform to their full potential or may be excluded from the match altogether. However, data on injury status is often difficult to obtain and may not be consistently available.

### 5. **Model Complexity and Overfitting**

The use of advanced neural networks, such as LSTM and GRU, while powerful, can lead to overfitting, especially with small datasets. The model might perform well on training data but fail to generalize to unseen data. This is particularly challenging when there are not enough features or data points to capture the variability in player performance.

### 6. **Assumption of Homogeneous Data**

The model assumes that the data is homogeneous and does not account for variations in data quality or missing values. In real-world scenarios, data may be incomplete or inconsistently recorded, which can affect model performance. The model could benefit from more sophisticated data preprocessing techniques to handle such issues.

### 7. **Simplified Fantasy Points Calculation**

The current fantasy points calculation logic is relatively simple and does not account for all possible scenarios in a cricket match. For example, it does not consider the impact of run-outs, catches, or stumpings, which can also contribute to fantasy points in various fantasy cricket leagues.

## Suggested Features for Enhanced Dataset

### **Player Features:**

1. **Runs Scored:** Total runs scored by the player across all matches.
2. **Wickets Taken:** Total number of wickets taken by the player.
3. **Balls Faced:** Total balls faced by the player in batting.
4. **Balls Bowled:** Total balls bowled by the player.
5. **Overs Bowled:** Total overs bowled by the player.
6. **Maidens Bowled:** Total maidens bowled by the player.
7. **Runs Conceded:** Total runs conceded by the player while bowling.
8. **Batting Average:** Average runs scored per dismissal.
9. **Bowling Average:** Average runs conceded per wicket taken.
10. **Strike Rate:** Runs scored per 100 balls faced.
11. **Economy Rate:** Runs conceded per over bowled.
12. **Centuries Scored:** Number of centuries scored by the player.
13. **Half-Centuries Scored:** Number of half-centuries scored by the player.
14. **Ducks Scored:** Number of times the player has been dismissed for zero runs.
15. **Wickets Taken in Last Match:** Number of wickets taken by the player in the most recent match.
16. **Runs Scored in Last Match:** Runs scored by the player in the most recent match.
17. **Player Age:** Age of the player.
18. **Player Experience:** Number of matches played by the player.
19. **Player Role:** Role of the player (e.g., batsman, bowler, all-rounder, wicket-keeper).
20. **Injury Status:** Fitness status of the player (e.g., fit, injured).

### **Team Features:**

1. **Wins:** Total number of wins by the team.
2. **Losses:** Total number of losses by the team.
3. **Draws:** Total number of draws by the team.
4. **Net Run Rate:** Overall run rate minus the opposition's run rate.
5. **Total Runs Scored:** Total runs scored by the team.
6. **Total Wickets Taken:** Total number of wickets taken by the team.
7. **Team Rating:** The rating of the team, such as ICC ranking.
8. **Home Wins:** Number of wins by the team at home grounds.
9. **Away Wins:** Number of wins by the team at away grounds.
10. **Home Losses:** Number of losses by the team at home grounds.
11. **Away Losses:** Number of losses by the team at away grounds.

### **Match Features:**

1. **Match Date:** Date of the match.
2. **Match Time:** Time when the match is played.
3. **Pitch Type:** Type of pitch (e.g., spinning, seaming).
4. **Weather Conditions:** Weather conditions during the match (e.g., sunny, cloudy).
5. **Match Format:** Format of the match (e.g., T20, ODI, Test).
6. **Match Importance:** Importance of the match (e.g., league match, playoff).
7. **Match Location:** Location of the match (e.g., home, away).

### **Opponent Features:**

1. **Opponent Team Rating:** Rating of the opposing team.
2. **Opponent Player Rating:** Ratings of key players in the opposing team.
3. **Head-to-Head Wins:** Number of wins against the opposing team in past encounters.
4. **Head-to-Head Losses:** Number of losses against the opposing team in past encounters.

### **Contextual Features:**

1. **Time of Year:** Time of the year (e.g., month).
2. **Tournament Stage:** Stage of the tournament (e.g., group stage, knockout stage).

## Conclusion

This project demonstrates how machine learning and neural networks can be used to predict cricket player performance and calculate fantasy points. By selecting a top 11-player team based on these predictions, the project provides a practical tool for fantasy cricket enthusiasts to enhance their team selection strategies. By addressing the current limitations and incorporating a more comprehensive set of features, the model can be significantly improved. A richer dataset with diverse features will enable the model to better capture the nuances of cricket matches and provide more accurate predictions for player performance and fantasy points. Future iterations of this project should focus on expanding the feature set, improving data preprocessing, and considering the inclusion of contextual and environmental factors.
