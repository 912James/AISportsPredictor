# app.py: Flask web application for predicting sports game outcomes
# Integrates data fetching, model training, and web serving with visualization capabilities

from flask import Flask, render_template, request
import pandas as pd
from predictor import ScoreFetcher, prepare_data, train_model, predict_game
import os
from datetime import datetime, timedelta
import json
import pytz

app = Flask(__name__)
fetcher = ScoreFetcher()
sports = ['nba', 'nfl', 'mlb']
data = {}
models = {}
encoders = {}
scalers = {}
static_avgs = {}
teams = {sport: fetcher.team_filters.get(sport, []) for sport in sports}  # Initialize with team_filters

def load_full_season_data(sport):
    csv_path = f'{sport}_data_2024_25.csv'
    season_start = {'nba': '20241022', 'nfl': '20240905', 'mlb': '20240320'}
    start_date_str = season_start.get(sport, '20240101')
    end_date = datetime.now(pytz.UTC)
    print(f"Loading {sport} data from {start_date_str} to {end_date.strftime('%Y%m%d')}", flush=True)

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {csv_path}: {df.shape}, columns: {df.columns.tolist()}", flush=True)
            if not df.empty and all(col in df.columns for col in ['date', 'team1', 'team2', 'score1', 'score2', 'winner']):
                last_date = pd.to_datetime(df['date']).max()
                if last_date < end_date - timedelta(days=1):
                    missing_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
                    print(f"Fetching {sport} updates from {missing_start} to {end_date.strftime('%Y%m%d')}", flush=True)
                    new_df = fetcher.fetch_season(sport, missing_start, end_date.strftime("%Y%m%d"))
                    if not new_df.empty:
                        df = pd.concat([df, new_df]).drop_duplicates().reset_index(drop=True)
                        df.to_csv(csv_path, index=False)
                        print(f"Updated {csv_path} with {len(df)} rows", flush=True)
                return df
        except Exception as e:
            print(f"Error loading {csv_path}: {e}, fetching last 7 days", flush=True)

    last_week = (end_date - timedelta(days=7)).strftime("%Y%m%d")
    print(f"Fetching {sport} from {last_week} to {end_date.strftime('%Y%m%d')}", flush=True)
    df = fetcher.fetch_season(sport, last_week, end_date.strftime("%Y%m%d"))
    if not df.empty:
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} with {len(df)} rows", flush=True)
    return df if not df.empty else pd.DataFrame(columns=['date', 'team1', 'team2', 'score1', 'score2', 'winner'])

# Initialize data and models for each sport at startup
for sport in sports:
    data[sport] = load_full_season_data(sport)
    # Update teams with fetched data if available
    if not data[sport].empty and 'team1' in data[sport].columns and 'team2' in data[sport].columns:
        teams[sport] = sorted(set(data[sport]['team1']).union(data[sport]['team2']))
        print(f"Updated teams for {sport} from data: {teams[sport]}", flush=True)
    else:
        print(f"No valid data for {sport}, retaining predefined team list: {teams[sport]}", flush=True)

    if data[sport].empty or 'winner' not in data[sport].columns or len(data[sport]['winner'].unique()) < 2 or len(data[sport]) < 10:
        print(f"Skipping model training for {sport}: insufficient data", flush=True)
        models[sport] = None
        continue

    prepared = prepare_data(data[sport])
    if len(prepared) == 5:
        X, y, le, avgs, scaler = prepared
        encoders[sport] = le
    else:
        X, y, avgs, scaler = prepared
        encoders[sport] = None
    models[sport] = train_model(X, y)
    scalers[sport] = scaler
    static_avgs[sport] = avgs
    print(f"Initialized teams for {sport}: {teams[sport]}", flush=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(f"Received POST data: {request.form}", flush=True)
        sport = request.form.get('sport', '')
        team1 = request.form.get('team1', '')
        team2 = request.form.get('team2', '')
        if not sport or not team1 or not team2:
            return render_template('index.html', sports=sports, teams=teams, error="Please select a sport and two teams", selected_sport=sport), 400
        if team1 == team2:
            return render_template('index.html', sports=sports, teams=teams, error="Teams must be different", selected_sport=sport)
        if models[sport] is None:
            return render_template('index.html', sports=sports, teams=teams, error=f"No model available for {sport}", selected_sport=sport)
        prediction = predict_game(models[sport], static_avgs[sport], team1, team2, scalers[sport])
        confidence = float(prediction.split('(')[1].split('%')[0])
        winner = prediction.split()[0]
        team1_prob = confidence if winner == team1 else 100 - confidence
        team2_prob = confidence if winner == team2 else 100 - confidence
        chart_data = {'team1': team1, 'team2': team2, 'team1_prob': team1_prob, 'team2_prob': team2_prob}
        return render_template('result.html', prediction=prediction, team1=team1, team2=team2, 
                             confidence=confidence, winner=winner, chart_data=json.dumps(chart_data), sport=sport)
    return render_template('index.html', sports=sports, teams=teams, selected_sport='nba')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)