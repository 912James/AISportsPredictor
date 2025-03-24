# app.py: Flask web app for predicting sports game outcomes
# This script powers a web server that fetches sports data, trains models, and serves predictions with charts.

from flask import Flask, render_template, request  # Core Flask tools for web app functionality
import pandas as pd  # For efficient data manipulation and storage
from predictor import ScoreFetcher, prepare_data, train_model, predict_game  # Custom functions from predictor.py
import os  # For file system operations like checking CSV existence
from datetime import datetime, timedelta  # For handling dates and time ranges
import json  # To format chart data for the frontend
import pytz  # For UTC timezone support, crucial for Heroku

# Initialize the Flask application
app = Flask(__name__)

# Instantiate a ScoreFetcher object to retrieve sports data from an API
fetcher = ScoreFetcher()

# List of sports supported by the app
sports = ['nba', 'nfl', 'mlb']

# Dictionaries to hold data and models for each sport
data = {}  # Stores raw game data as DataFrames
models = {}  # Holds trained machine learning models
encoders = {}  # Stores label encoders (if used, currently optional)
scalers = {}  # Stores feature scalers for consistent predictions
static_avgs = {}  # Stores average team performance scores
teams = {}  # Stores sorted lists of team names per sport

# Function to load or fetch season data for a given sport
def load_full_season_data(sport):
    csv_path = f'{sport}_data_2024_25.csv'  # File path for storing/retrieving season data
    # Define season start dates for each sport
    season_start = {'nba': '20241022', 'nfl': '20240905', 'mlb': '20240320'}
    start_date_str = season_start.get(sport, '20240101')  # Default to Jan 1 if sport not found
    end_date = datetime.now(pytz.UTC)  # Get current UTC date for fresh data
    print(f"Starting {sport} data load from {start_date_str} to {end_date.strftime('%Y%m%d')}", flush=True)

    # Check if a CSV file exists for this sport
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)  # Load existing data
            print(f"Loaded {csv_path}: {df.shape}, columns: {df.columns.tolist()}", flush=True)
            # Verify the data has all required columns
            if not df.empty and all(col in df.columns for col in ['date', 'team1', 'team2', 'score1', 'score2', 'winner']):
                last_date = pd.to_datetime(df['date']).max()  # Find the latest date in the data
                # If data is stale, fetch updates
                if last_date < end_date - timedelta(days=1):
                    missing_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
                    print(f"Fetching {sport} from {missing_start} to {end_date.strftime('%Y%m%d')}", flush=True)
                    new_df = fetcher.fetch_season(sport, missing_start, end_date.strftime("%Y%m%d"))
                    if not new_df.empty:
                        df = pd.concat([df, new_df]).drop_duplicates().reset_index(drop=True)  # Merge and deduplicate
                        df.to_csv(csv_path, index=False)  # Save updated data
                        print(f"Updated {csv_path} with {len(df)} rows", flush=True)
                return df  # Return the loaded or updated DataFrame
        except Exception as e:
            print(f"Error with {csv_path}: {e}, fetching last 7 days", flush=True)  # Log errors and fallback

    # Fallback: fetch last 7 days if no valid CSV
    last_week = (end_date - timedelta(days=7)).strftime("%Y%m%d")
    print(f"Fetching {sport} from {last_week} to {end_date.strftime('%Y%m%d')}", flush=True)
    df = fetcher.fetch_season(sport, last_week, end_date.strftime("%Y%m%d"))
    if not df.empty:
        df.to_csv(csv_path, index=False)  # Save fresh data
        print(f"Saved {csv_path} with {len(df)} rows", flush=True)
    # Return data or an empty DataFrame with expected columns if fetch fails
    return df if not df.empty else pd.DataFrame(columns=['date', 'team1', 'team2', 'score1', 'score2', 'winner'])

# Initialize data and models for all sports when the app starts
for sport in sports:
    data[sport] = load_full_season_data(sport)  # Load or fetch game data
    # Check if data is sufficient for training
    if data[sport].empty or 'winner' not in data[sport].columns or len(data[sport]['winner'].unique()) < 2 or len(data[sport]) < 10:
        print(f"Skipping {sport}: insufficient data", flush=True)  # Skip if data lacks winners or variety
        models[sport] = None
        teams[sport] = []
        continue
    prepared = prepare_data(data[sport])  # Prepare data for machine learning
    # Handle varying return values from prepare_data
    if len(prepared) == 5:
        X, y, le, avgs, scaler = prepared  # Unpack with encoder if present
        encoders[sport] = le  # Store encoder
    else:
        X, y, avgs, scaler = prepared  # Unpack without encoder
        encoders[sport] = None  # No encoder used
    models[sport] = train_model(X, y)  # Train a model for this sport
    scalers[sport] = scaler  # Save scaler for predictions
    static_avgs[sport] = avgs  # Save team averages
    teams[sport] = sorted(set(data[sport]['team1']).union(data[sport]['team2']))  # Get unique teams
    print(f"Teams for {sport}: {teams[sport]}", flush=True)  # Log teams for debugging

# Define the main route handling both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(f"POST data: {request.form}", flush=True)  # Log incoming form data for debugging
        # Safely extract form inputs with defaults
        sport = request.form.get('sport', '')
        team1 = request.form.get('team1', '')
        team2 = request.form.get('team2', '')
        # Input validation
        if not sport or not team1 or not team2:
            return render_template('index.html', sports=sports, teams=teams, error="Please select a sport and two teams", selected_sport=sport), 400  # Bad request
        if team1 == team2:
            return render_template('index.html', sports=sports, teams=teams, error="Pick two different teams!", selected_sport=sport)  # Same team error
        if models[sport] is None:
            return render_template('index.html', sports=sports, teams=teams, error=f"No model for {sport}", selected_sport=sport)  # No model available
        # Generate prediction using the trained model
        prediction = predict_game(models[sport], static_avgs[sport], team1, team2, scalers[sport])
        confidence = float(prediction.split('(')[1].split('%')[0])  # Parse confidence percentage
        winner = prediction.split()[0]  # Extract winning team
        # Calculate probabilities for both teams for the chart
        team1_prob = confidence if winner == team1 else 100 - confidence
        team2_prob = confidence if winner == team2 else 100 - confidence
        chart_data = {'team1': team1, 'team2': team2, 'team1_prob': team1_prob, 'team2_prob': team2_prob}  # Structure data for chart
        # Render the result page with prediction and chart data
        return render_template('result.html', prediction=prediction, team1=team1, team2=team2, 
                             confidence=confidence, winner=winner, chart_data=json.dumps(chart_data), sport=sport)
    # For GET requests, render the input form with NBA as default
    return render_template('index.html', sports=sports, teams=teams, selected_sport='nba')

# Entry point for running the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Get port from environment (Heroku) or default to 5001
    app.run(host='0.0.0.0', port=port, debug=False)  # Start the server, accessible externally