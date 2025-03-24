# predictor.py: Core logic for fetching ESPN sports data and training prediction models

import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

class ScoreFetcher:
    def __init__(self):
        # Define base URL for ESPN API and headers to mimic a browser request
        self.api_base_url = "https://site.api.espn.com/apis/site/v2/sports"
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'}
        # Map sports to their respective API endpoints
        self.league_paths = {'nba': 'basketball/nba', 'nfl': 'football/nfl', 'mlb': 'baseball/mlb'}
        # Define valid team abbreviations for filtering relevant games
        self.team_filters = {
            'nba': {'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GS', 'HOU', 'IND', 
                    'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NO', 'NY', 'OKC', 'ORL', 'PHI', 'PHX', 
                    'POR', 'SAC', 'SA', 'TOR', 'UTAH', 'WSH'},
            'nfl': {'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 
                    'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 
                    'LV', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WSH'},
            'mlb': {'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KC', 
                    'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SF', 
                    'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH'}
        }

    def fetch_season(self, sport='nba', start_date=None, end_date=None):
        # Validate sport input against supported leagues
        if sport not in self.league_paths:
            raise ValueError(f"Sport {sport} not supported.")
        # Set default season start dates if not provided
        start_dates = {'nba': '20241022', 'nfl': '20240905', 'mlb': '20240320'}
        start_date = start_date or start_dates.get(sport, '20240101')
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        games = []
        current_date = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        # Iterate through dates to fetch game data
        while current_date <= end:
            date_str = current_date.strftime("%Y%m%d")
            url = f"{self.api_base_url}/{self.league_paths[sport]}/scoreboard?dates={date_str}&limit=500"
            print(f"Fetching {sport} data for {date_str}...", flush=True)
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                for event in data.get('events', []):
                    if event['status']['type']['completed']:
                        game = {}
                        comp = event['competitions'][0]
                        teams = comp['competitors']
                        if len(teams) != 2:
                            continue
                        team1 = teams[0]['team']['abbreviation']
                        team2 = teams[1]['team']['abbreviation']
                        # Filter games to include only specified teams
                        if sport in self.team_filters and (team1 not in self.team_filters[sport] or team2 not in self.team_filters[sport]):
                            continue
                        game['date'] = event['date']
                        game['team1'] = team1
                        game['team2'] = team2
                        game['score1'] = int(teams[0]['score'])
                        game['score2'] = int(teams[1]['score'])
                        game['winner'] = 1 if game['score1'] > game['score2'] else 0
                        games.append(game)
            except Exception as e:
                print(f"Error fetching data for {date_str}: {e}", flush=True)
            current_date += timedelta(days=1)
        df = pd.DataFrame(games)
        print(f"Fetched {sport} data: {df.shape}, columns: {df.columns.tolist()}", flush=True)
        if not df.empty:
            df.to_csv(f'{sport}_data_2024_25.csv', index=False)
            print(f"Saved {sport}_data_2024_25.csv with {len(df)} rows", flush=True)
        return df

    def fetch_team_averages(self, sport='nba', days_back=10):
        team_scores = {}
        today = datetime.now()
        start_date = (today - timedelta(days=days_back)).strftime("%Y%m%d")
        end_date = today.strftime("%Y%m%d")
        url = f"{self.api_base_url}/{self.league_paths[sport]}/scoreboard?dates={start_date}-{end_date}&limit=500"
        print(f"Fetching team averages for {sport} from {start_date} to {end_date}", flush=True)
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            for event in data.get('events', []):
                if event['status']['type']['completed']:
                    comp = event['competitions'][0]
                    teams = comp['competitors']
                    if len(teams) != 2:
                        continue
                    team1, score1 = teams[0]['team']['abbreviation'], int(teams[0]['score'])
                    team2, score2 = teams[1]['team']['abbreviation'], int(teams[1]['score'])
                    if sport in self.team_filters and (team1 not in self.team_filters[sport] or team2 not in self.team_filters[sport]):
                        continue
                    # Aggregate scores for each team
                    team_scores[team1] = team_scores.get(team1, []) + [score1]
                    team_scores[team2] = team_scores.get(team2, []) + [score2]
        except Exception as e:
            print(f"Error fetching {sport} data: {e}", flush=True)
            return None
        # Calculate average scores per team
        team_avgs = {team: sum(scores) / len(scores) for team, scores in team_scores.items() if scores}
        print(f"Team averages for {sport}: {team_avgs}", flush=True)
        return team_avgs

def prepare_data(df):
    le = LabelEncoder()
    all_teams = pd.concat([df['team1'], df['team2']]).unique()
    le.fit(all_teams)
    # Encode team names for model compatibility
    df['team1_encoded'] = le.transform(df['team1'])
    df['team2_encoded'] = le.transform(df['team2'])
    team_avgs = {}
    for team in all_teams:
        team_scores = pd.concat([df[df['team1'] == team]['score1'], df[df['team2'] == team]['score2']])
        team_avgs[team] = team_scores.mean() if not team_scores.empty else 0
    df['team1_avg'] = df['team1'].map(team_avgs)
    df['team2_avg'] = df['team2'].map(team_avgs)
    # Compute difference in team averages as a feature
    df['avg_diff'] = df['team1_avg'] - df['team2_avg']
    X = df[['avg_diff']]
    y = df['winner']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize features for model training
    print(f"Team averages: {team_avgs}", flush=True)
    print(f"Recent avg_diff: {df['avg_diff'].tail(5).tolist()}", flush=True)
    print(f"Scaled features: {X_scaled[-5:].tolist()}", flush=True)
    print(f"Target values: {y.tail(5).tolist()}", flush=True)
    return X_scaled, y, le, team_avgs, scaler

def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)  # Train logistic regression model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}", flush=True)
    print(f"Model coefficients: {model.coef_[0][0]}, intercept: {model.intercept_[0]}", flush=True)
    return model

def predict_game(model, avgs, team1, team2, scaler):
    team1_avg = avgs.get(team1, 0)
    team2_avg = avgs.get(team2, 0)
    avg_diff = team1_avg - team2_avg
    X_new = scaler.transform([[avg_diff]])  # Scale input for prediction
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][pred] * 100  # Calculate prediction probability
    winner = team1 if pred == 1 else team2
    return f"{winner} wins ({prob:.1f}%)"
