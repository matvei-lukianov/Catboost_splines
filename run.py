from orderbook import *
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.metrics import r2_score

start_timestamp = 1725580800014
end_timestamp = 1725583800014  # 3000 seconds later
file_path = "merged_logs_1_binance_BTC-USDT_2024-09-06.jsonl"

df_orderbook = create_orderbook_dataframe(file_path, start_timestamp, end_timestamp)

# Example usage
weights_a = np.ones(10)  
weights_b = np.ones(10)

# Set the start and end of the strip (e.g., 25th to 75th percentile)
alpha_start = 0.0
alpha_end = 0.1

# Compute all indicators for the order book DataFrame
indicators_df = compute_indicators(df_orderbook, weights_a, weights_b, alpha_start, alpha_end)

def X_y_compute(df_orderbook, df_indicators, time_column='timestamp', seconds_ahead=1):
    """
    Calculate the return after a given number of seconds.
    
    Args:
        df (pd.DataFrame): Dataframe with price data.
        time_column (str): The name of the column with the timestamp.
        seconds_ahead (int): The number of seconds ahead to calculate the return.
    
    Returns:
        pd.DataFrame: DataFrame with the target return column added.
    """
    df = df_orderbook
    # Calculate the mid price
    df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
    
    # Calculate the return after `seconds_ahead`
    df['mid_price_future'] = df['mid_price'].shift(-seconds_ahead)
    
    # Calculate the percentage return
    df['return_after_n_seconds'] = (df['mid_price_future'] - df['mid_price']) / df['mid_price']
    
    # Drop the 'mid_price_future' column as it's no longer needed
    df = df.drop(columns=['mid_price_future'])
    X = indicators_df.iloc[:-10]
    y = df['return_after_n_seconds'].dropna()
    
    return X, y


X, y = X_y_compute(df_orderbook, indicators_df, seconds_ahead = 10)

X.to_csv('X.csv')
y.to_csv('y.csv')


# # Define window size for in-sample and out-sample
# in_sample_size = 1000
# out_sample_size = 100
# step_size = 10

# # Lists to store in-sample and OOS predictions and true values
# oos_predictions_catboost = []
# oos_true_values = []
# in_sample_mae_catboost_list = []
# in_sample_r2_catboost_list = []
# oos_mae_catboost_list = []
# oos_r2_catboost_list = []

# oos_predictions_linear = []
# in_sample_mae_linear_list = []
# in_sample_r2_linear_list = []
# oos_mae_linear_list = []
# oos_r2_linear_list = []

# from tqdm import tqdm
# catboost_reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0)
# linear_reg = LinearRegression()
# # Loop over the dataset using rolling window
# for start in tqdm(range(0, len(X) - in_sample_size - out_sample_size, step_size)):
#     # Define in-sample and out-sample ranges
#     end_in_sample = start + in_sample_size
#     end_out_sample = end_in_sample + out_sample_size
    
#     # Split data into in-sample and out-sample
#     X_train = X.iloc[start:end_in_sample]
#     y_train = y.iloc[start:end_in_sample]
    
#     X_test = X.iloc[end_in_sample:end_out_sample]
#     y_test = y.iloc[end_in_sample:end_out_sample]
    
#     # --- CatBoost Regressor ---
#     # Train CatBoost Regressor on the in-sample data
#     catboost_reg.fit(X_train, y_train)
    
#     # In-sample predictions and metrics
#     y_pred_train_catboost = catboost_reg.predict(X_train)
#     in_sample_mae_catboost = mean_absolute_error(y_train, y_pred_train_catboost)
#     in_sample_r2_catboost = r2_score(y_train, y_pred_train_catboost)
#     in_sample_mae_catboost_list.append(in_sample_mae_catboost)
#     in_sample_r2_catboost_list.append(in_sample_r2_catboost)
    
#     # Out-of-sample predictions and metrics
#     y_pred_test_catboost = catboost_reg.predict(X_test)
#     oos_predictions_catboost.extend(y_pred_test_catboost)
#     oos_true_values.extend(y_test)
#     oos_mae_catboost = mean_absolute_error(y_test, y_pred_test_catboost)
#     oos_r2_catboost = r2_score(y_test, y_pred_test_catboost)
#     oos_mae_catboost_list.append(oos_mae_catboost)
#     oos_r2_catboost_list.append(oos_r2_catboost)

#     # --- Linear Regressor ---
#     # Train Linear Regressor on the in-sample data
#     linear_reg.fit(X_train, y_train)
    
#     # In-sample predictions and metrics
#     y_pred_train_linear = linear_reg.predict(X_train)
#     in_sample_mae_linear = mean_absolute_error(y_train, y_pred_train_linear)
#     in_sample_r2_linear = r2_score(y_train, y_pred_train_linear)
#     in_sample_mae_linear_list.append(in_sample_mae_linear)
#     in_sample_r2_linear_list.append(in_sample_r2_linear)
    
#     # Out-of-sample predictions and metrics
#     y_pred_test_linear = np.minimum(linear_reg.predict(X_test), 0.001)
#     oos_predictions_linear.extend(y_pred_test_linear)
#     oos_mae_linear = mean_absolute_error(y_test, y_pred_test_linear)
#     oos_r2_linear = r2_score(y_test, y_pred_test_linear)
#     oos_mae_linear_list.append(oos_mae_linear)
#     oos_r2_linear_list.append(oos_r2_linear)

# # Now calculate the average In-sample and OOS metrics for each model

# # --- CatBoost Performance ---
# mean_in_sample_mae_catboost = np.mean(in_sample_mae_catboost_list)
# mean_in_sample_r2_catboost = np.mean(in_sample_r2_catboost_list)
# mean_oos_mae_catboost = np.mean(oos_mae_catboost_list)
# mean_oos_r2_catboost = np.mean(oos_r2_catboost_list)

# print(f"CatBoost Regressor:")
# print(f"Mean In-sample MAE: {mean_in_sample_mae_catboost}")
# print(f"Mean In-sample R²: {mean_in_sample_r2_catboost}")
# print(f"Mean Out-of-sample MAE: {mean_oos_mae_catboost}")
# print(f"Mean Out-of-sample R²: {mean_oos_r2_catboost}")

# # --- Linear Regressor Performance ---
# mean_in_sample_mae_linear = np.mean(in_sample_mae_linear_list)
# mean_in_sample_r2_linear = np.mean(in_sample_r2_linear_list)
# mean_oos_mae_linear = np.mean(oos_mae_linear_list)
# mean_oos_r2_linear = np.mean(oos_r2_linear_list)

# print(f"Linear Regressor:")
# print(f"Mean In-sample MAE: {mean_in_sample_mae_linear}")
# print(f"Mean In-sample R²: {mean_in_sample_r2_linear}")
# print(f"Mean Out-of-sample MAE: {mean_oos_mae_linear}")
# print(f"Mean Out-of-sample R²: {mean_oos_r2_linear}")