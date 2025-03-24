# AI Sports Predictor

AI Sports Predictor is a web application that leverages machine learning and live sports data from ESPN to predict the winner of a game between two teams, providing a confidence percentage for the outcome. Built with Python, Flask, and scikit-learn, this project fetches real-time and historical game data, trains a logistic regression model, and visualizes predictions using interactive charts.

## Features
Real-Time Data Fetching: Retrieves live and historical game scores from ESPN's API for NBA, NFL, and MLB.

Predictive Modeling: Uses logistic regression to predict game winners based on team performance averages.

User Interface: Offers an intuitive web form to select a sport and two teams, displaying predictions with a bar chart of win probabilities.

Data Persistence: Stores fetched data in CSV files for efficient reuse and updates.

Deployment: Hosted on Heroku for public access and demonstration.

## Demo
Try the live application here: AI Sports Predictor on Heroku
sports-predictor-2c313aca4669.herokuapp.com

## How It Works
Data Collection: The ScoreFetcher class queries ESPN's API to gather game data (team names, scores, dates) for a specified sport and season.

Data Processing: Historical data is processed to compute team scoring averages, which are used as features for prediction.

Model Training: A logistic regression model is trained on the difference in team averages to predict the winner (Team 1 or Team 2).

Prediction: Users select a sport and two teams via a web form; the model predicts the winner and confidence percentage.

Visualization: Results are displayed with a Chart.js bar graph showing win probabilities for both teams.




