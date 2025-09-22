import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from flasgger import Swagger
import mysql.connector
import traceback
from datetime import datetime
from pandas.tseries.offsets import MonthBegin
from dotenv import load_dotenv
import os

app = Flask(__name__)
swagger = Swagger(app)
load_dotenv()

# MySQL connection setup using environment variables
try:
    connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )
    print("MySQL connection established successfully.")
except mysql.connector.Error as err:
    print(f"Error connecting to MySQL: {err}")
    connection = None
    exit(1)

# Load data from tbl_items
try:
    if connection:
        print("Loading item data from tbl_items...")
        query_items = """
        SELECT ItemID, Item_Name 
        FROM tbl_items;
        """
        items_data = pd.read_sql(query_items, connection)
        items_data['Item_Name'] = items_data['Item_Name'].str.strip().str.lower().str.rstrip('.')
        items_data = items_data[['ItemID', 'Item_Name']].dropna()
        print("Item data loaded successfully.")
except Exception as e:
    print(f"Error loading tbl_items data: {e}")
    items_data = pd.DataFrame()
    exit(1)

# Load data from tbl_pharmacyissuedetails
try:
    if connection:
        print("Loading pharmacy issue data from tbl_pharmacyissuedetails...")
        query_pharmacy = """
        SELECT IssueDate, IssueQuantity, ItemID 
        FROM tbl_pharmacyissuedetails;
        """
        pharmacy_data = pd.read_sql(query_pharmacy, connection)
        pharmacy_data['IssueDate'] = pd.to_datetime(pharmacy_data['IssueDate'], errors='coerce', dayfirst=True)
        pharmacy_data = pharmacy_data[['IssueDate', 'IssueQuantity', 'ItemID']].dropna()
        print("Pharmacy issue data loaded successfully.")
except Exception as e:
    print(f"Error loading tbl_pharmacyissuedetails data: {e}")
    pharmacy_data = pd.DataFrame()
    exit(1)

# Preprocess the data
try:
    print("Preprocessing data...")
    # Preprocess items_data
    items_data['ItemID'] = items_data['ItemID'].astype(int)
    items_data = items_data.drop_duplicates(subset=['ItemID'])
    items_data.reset_index(drop=True, inplace=True)

    # Preprocess pharmacy_data
    pharmacy_data['IssueQuantity'] = pd.to_numeric(pharmacy_data['IssueQuantity'], errors='coerce')
    pharmacy_data = pharmacy_data.dropna(subset=['IssueQuantity'])
    pharmacy_data['ItemID'] = pharmacy_data['ItemID'].astype(int)

    # Merge pharmacy_data with items_data to get Item_Name for predictions
    data = pharmacy_data.merge(items_data[['ItemID', 'Item_Name']], on='ItemID', how='left')
    data = data.dropna(subset=['Item_Name'])

    # Data for XGBoost (monthly aggregation)
    data['ds'] = data['IssueDate'].dt.to_period('M').apply(lambda r: r.start_time)
    agg_data = data.groupby('ds')['IssueQuantity'].sum().reset_index()
    item_data = data.groupby(['Item_Name', 'ds'])['IssueQuantity'].sum().reset_index()

    # Data for LSTM (daily pivoted data)
    pivot_data = data.groupby(['IssueDate', 'Item_Name'])['IssueQuantity'].sum().unstack(fill_value=0)
    pivot_data.columns = [col.strip().lower().rstrip('.') for col in pivot_data.columns]

    # Scale data for LSTM
    scalers = {}
    for item in pivot_data.columns:
        scaler = MinMaxScaler()
        pivot_data[item] = scaler.fit_transform(pivot_data[[item]])
        scalers[item] = scaler

    print("Data preprocessed successfully.")
except Exception as e:
    print(f"Error preprocessing data: {e}")
    agg_data = pd.DataFrame()
    item_data = pd.DataFrame()
    pivot_data = pd.DataFrame()
    scalers = {}
    exit(1)

# Load models
xgboost_model = joblib.load('xgboost_model.joblib')
print("XGBoost model loaded successfully.")
lstm_model = load_model('lstm_stock_prediction.h5', custom_objects={'mse': MeanSquaredError})
print("LSTM model loaded successfully.")

# XGBoost helper functions
def create_features(df):
    df = df.copy()
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
    for lag in [1, 2, 12]:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['rolling_mean_6'] = df['y'].shift(1).rolling(window=6).mean()
    return df

def predict_xgboost(item_name, year, month=None):
    # Prepare historical total data with dynamic date range
    start_date = pharmacy_data['IssueDate'].min().to_period('M').start_time
    end_date = pharmacy_data['IssueDate'].max().to_period('M').end_time
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    full_data = pd.DataFrame({'ds': date_range})
    full_data['ds'] = full_data['ds'].dt.to_period('M').apply(lambda r: r.start_time)
    full_data = full_data.merge(agg_data[['ds', 'IssueQuantity']], on='ds', how='left').fillna({'IssueQuantity': 0})
    full_data = full_data.rename(columns={'IssueQuantity': 'y'})
    
    full_data = create_features(full_data)
    full_data = full_data.fillna({
        'lag_1': 0, 'lag_2': 0, 'lag_12': 0,
        'rolling_mean_6': 0
    })
    
    # Generate future dates
    if month is None:
        start_date = pd.to_datetime(f'{year}-01-01')
        end_date = pd.to_datetime(f'{year}-12-31')
    else:
        start_date = pd.to_datetime(f'{year}-{month}-01')
        end_date = start_date
    
    if start_date <= full_data['ds'].max():
        return {"error": f"Prediction date must be after last historical date ({full_data['ds'].max().date()})"}
    
    future_dates = pd.date_range(start=full_data['ds'].max() + pd.offsets.MonthBegin(1), end=end_date, freq='MS')
    future_data = pd.DataFrame({'ds': future_dates})
    future_data['ds'] = future_data['ds'].dt.to_period('M').apply(lambda r: r.start_time)
    
    # Feature engineering for future data
    future_data['month'] = future_data['ds'].dt.month
    future_data['quarter'] = future_data['ds'].dt.quarter
    future_data['is_quarter_end'] = future_data['ds'].dt.is_quarter_end.astype(int)
    
    # Initialize lags
    last_12_months = full_data.tail(12).copy()
    predictions = []
    
    # Autoregressive prediction
    for date in future_dates:
        temp_df = future_data[future_data['ds'] == date].copy()
        if len(predictions) == 0:
            initial_lags = last_12_months[['lag_1', 'lag_2', 'lag_12', 'rolling_mean_6']].iloc[-1].values
            temp_df['lag_1'] = initial_lags[0]
            temp_df['lag_2'] = initial_lags[1]
            temp_df['lag_12'] = initial_lags[2]
            temp_df['rolling_mean_6'] = initial_lags[3]
        else:
            prev_preds = [p['yhat'] for p in predictions[-12:]] + [0] * (12 - len(predictions[-12:]))
            temp_df['lag_1'] = predictions[-1]['yhat']
            temp_df['lag_2'] = predictions[-2]['yhat'] if len(predictions) > 1 else 0
            temp_df['lag_12'] = prev_preds[-12] if len(prev_preds) >= 12 else 0
            temp_df['rolling_mean_6'] = sum(prev_preds[-6:]) / 6 if len(prev_preds) >= 6 else 0
        
        features = ['month', 'quarter', 'is_quarter_end', 'lag_1', 'lag_2', 'lag_12', 'rolling_mean_6']
        X_future = temp_df[features]
        pred = xgboost_model.predict(X_future)[0]
        predictions.append({'ds': date, 'yhat': pred})
    
    # Calculate item-specific proportions
    item_historical = item_data[item_data['Item_Name'] == item_name]
    item_historical = item_historical.merge(full_data[['ds', 'y']], on='ds', how='left')
    item_historical['month'] = item_historical['ds'].dt.month
    item_historical['proportion'] = item_historical['IssueQuantity'] / item_historical['y']
    
    final_predictions = []
    for pred in predictions:
        pred_month = pred['ds'].month
        pred_year = pred['ds'].year
        total_pred = pred['yhat']
        
        historical_avg_proportion = item_historical[item_historical['month'] == pred_month]['proportion'].mean()
        if pd.isna(historical_avg_proportion) or historical_avg_proportion == 0:
            num_items = len(data['Item_Name'].unique())
            item_pred = total_pred / num_items
        else:
            item_pred = total_pred * historical_avg_proportion
        
        final_predictions.append({
            'Year': int(pred_year),
            'Month': int(pred_month),
            'Predicted_IssueQuantity': float(item_pred)
        })
    
    if month is None:
        return {'Item_Name': item_name, 'Predictions': final_predictions}
    else:
        return {
            'Item_Name': item_name,
            'Year': int(year),
            'Month': int(month),
            'Predicted_IssueQuantity': float(final_predictions[-1]['Predicted_IssueQuantity'])
        }

# LSTM helper functions
def predict_lstm(item_name, future_date, forecast_type):
    print(f"Predicting with LSTM for item: {item_name}, date: {future_date}, type: {forecast_type}")
    if item_name not in pivot_data.columns:
        print(f"Item '{item_name}' not found in pivot_data columns: {pivot_data.columns}")
        return {"error": f"Item '{item_name}' not found."}
    
    scaler = scalers[item_name]
    historical_data = pivot_data[item_name].values.reshape(-1, 1)
    print(f"Historical data shape for {item_name}: {historical_data.shape}")
    
    historical_data_scaled = scaler.transform(historical_data)
    print(f"Scaled historical data shape: {historical_data_scaled.shape}")
    
    seq_length = 30
    n_features = pivot_data.shape[1]
    print(f"Sequence length: {seq_length}, Number of features: {n_features}")
    
    input_sequence = np.zeros((1, seq_length, n_features))
    item_idx = pivot_data.columns.get_loc(item_name)
    print(f"Item index for {item_name}: {item_idx}")
    
    if len(historical_data_scaled) < seq_length:
        print(f"Warning: Historical data length ({len(historical_data_scaled)}) is less than sequence length ({seq_length}). Padding with zeros.")
        padded_data = np.zeros((seq_length, 1))
        padded_data[-len(historical_data_scaled):] = historical_data_scaled
        input_sequence[0, :, item_idx] = padded_data.flatten()
    else:
        input_sequence[0, :, item_idx] = historical_data_scaled[-seq_length:].flatten()
    
    # Calculate days to predict
    last_trained_date = pivot_data.index[-1]
    future_days = (future_date - last_trained_date).days
    print(f"Last trained date: {last_trained_date}, Future days: {future_days}")
    
    if forecast_type == "day":
        start_date = future_date.replace(day=1)
        end_date = start_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        days_in_month = (end_date - start_date).days + 1
    else:
        start_date = future_date
        end_date = start_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        days_in_month = (end_date - start_date).days + 1
    print(f"Days in month to predict: {days_in_month}")
    
    predictions = []
    current_sequence = input_sequence.copy()
    
    for i in range(days_in_month):
        pred_scaled = lstm_model.predict(current_sequence, verbose=0)[0]
        print(f"Prediction iteration {i+1}/{days_in_month}, Scaled prediction: {pred_scaled[item_idx]}")
        pred_actual = scaler.inverse_transform([pred_scaled])[0][item_idx]
        print(f"Actual prediction: {pred_actual}")
        predictions.append(pred_actual)
        
        current_sequence = np.roll(current_sequence, shift=-1, axis=1)
        current_sequence[0, -1, item_idx] = pred_scaled[item_idx]
    
    monthly_total = np.sum(predictions)
    print(f"Monthly total predicted quantity: {monthly_total}")
    
    return {
        "Item_Name": item_name,
        "Forecast_Type": forecast_type,
        "Year": int(future_date.year),
        "Month": int(future_date.month),
        "Predicted_IssueQuantity": float(monthly_total)
    }

def predict_lstm_monthly(item_name, future_month):
    print(f"Predicting monthly with LSTM for item: {item_name}, month: {future_month}")
    return predict_lstm(item_name, future_month, "month")

def predict_lstm_yearly(item_name, future_year):
    print(f"Predicting yearly with LSTM for item: {item_name}, year: {future_year}")
    predictions = []
    for month in range(1, 13):
        try:
            month_start = pd.to_datetime(f"{future_year.year}-{month}-01")
            print(f"Predicting for month {month}: {month_start}")
            monthly_pred = predict_lstm(item_name, month_start, "month")
            if "error" in monthly_pred:
                return monthly_pred
            predictions.append({
                "Year": int(monthly_pred["Year"]),
                "Month": int(monthly_pred["Month"]),
                "Predicted_IssueQuantity": float(monthly_pred["Predicted_IssueQuantity"])
            })
        except Exception as e:
            print(f"Error predicting for month {month}: {e}")
            raise
    
    return {
        "Item_Name": item_name,
        "Predictions": predictions
    }

# Combined prediction endpoint
@app.route('/predict', methods=['GET'])
def predict():
    """
    Predict future issuance for a given item using XGBoost or LSTM.
    ---
    parameters:
      - name: item_name
        in: query
        type: string
        required: true
        description: Name of the item (case-insensitive)
      - name: future_date
        in: query
        type: string
        required: true
        description: Future date in format YYYY-MM-DD, YYYY-MM, or YYYY
      - name: model
        in: query
        type: string
        required: true
        description: Model to use for prediction ('xgboost' or 'lstm')
    responses:
      200:
        description: Prediction result
      400:
        description: Invalid input
      500:
        description: Server error
    """
    try:
        item_name = request.args.get('item_name', '').strip().lower().rstrip('.')
        future_date_str = request.args.get('future_date', '')
        model_type = request.args.get('model', '').lower()

        print(f"Received request: item_name={item_name}, future_date={future_date_str}, model={model_type}")

        if not item_name or not future_date_str or not model_type:
            return jsonify({"error": "Provide 'item_name', 'future_date', and 'model'."}), 400

        # Parse future date
        if len(future_date_str) == 10:
            future_date = pd.to_datetime(future_date_str, errors='coerce')
            forecast_type = "day"
        elif len(future_date_str) == 7:
            future_date = pd.to_datetime(future_date_str + "-01", errors='coerce')
            forecast_type = "month"
        elif len(future_date_str) == 4:
            future_date = pd.to_datetime(future_date_str + "-01-01", errors='coerce')
            forecast_type = "year"
        else:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD, YYYY-MM, or YYYY."}), 400

        if pd.isnull(future_date):
            return jsonify({"error": "Invalid date format."}), 400

        print(f"Parsed future_date: {future_date}, forecast_type: {forecast_type}")

        # Validate date for XGBoost
        if model_type == "xgboost" and forecast_type == "day":
            return jsonify({"error": "XGBoost model only supports monthly or yearly predictions."}), 400

        # Check if future date is after last trained date
        last_trained_date = pivot_data.index[-1]
        future_days = (future_date - last_trained_date).days
        print(f"Last trained date: {last_trained_date}, Future days: {future_days}")
        if future_days <= 0:
            return jsonify({"error": f"Future date must be after last trained date ({last_trained_date.date()})."}), 400

        # Make prediction based on model
        if model_type == "xgboost":
            year = future_date.year
            month = future_date.month if forecast_type == "month" else None
            print(f"Calling predict_xgboost with year={year}, month={month}")
            result = predict_xgboost(item_name, year, month)
        elif model_type == "lstm":
            if forecast_type == "day":
                print("Calling predict_lstm for daily forecast")
                result = predict_lstm(item_name, future_date, "day")
            elif forecast_type == "month":
                print("Calling predict_lstm_monthly")
                result = predict_lstm_monthly(item_name, future_date)
            elif forecast_type == "year":
                print("Calling predict_lstm_yearly")
                result = predict_lstm_yearly(item_name, future_date)
        else:
            return jsonify({"error": "Invalid model type. Use 'xgboost' or 'lstm'."}), 400

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)