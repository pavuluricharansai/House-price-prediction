import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify, render_template
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Data Collection
def load_data():
    try:
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['MedHouseVal'] = housing.target
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Step 2: Data Preprocessing
def preprocess_data(df):
    try:
        df = df.dropna()
        features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        X = df[features]
        y = df['MedHouseVal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Data preprocessed successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

# Step 3: Model Training
def train_model(X_train, y_train, X_test, y_test):
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Model trained - MSE: {mse:.4f}, R2: {r2:.4f}")
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

# Step 4: Visualization
def create_visualizations(df, model, X_test, y_test, features):
    try:
        os.makedirs('static', exist_ok=True)
        # Feature importance plot
        importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('static/feature_importance.png')
        plt.close()
        # Actual vs Predicted plot
        y_pred = model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted House Prices')
        plt.tight_layout()
        plt.savefig('static/actual_vs_predicted.png')
        plt.close()
        logger.info("Visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise

# Step 5: Save Model and Scaler
def save_artifacts(model, scaler):
    try:
        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Model and scaler saved successfully")
    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}")
        raise

# Step 6: Flask App
app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('model.pkl') if os.path.exists('model.pkl') else None
    scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    model, scaler = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500
    try:
        data = request.get_json()
        features = [
            float(data['MedInc']),
            float(data['HouseAge']),
            float(data['AveRooms']),
            float(data['AveBedrms']),
            float(data['Population']),
            float(data['AveOccup']),
            float(data['Latitude']),
            float(data['Longitude'])
        ]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        logger.info(f"Prediction made: {prediction:.2f}")
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Step 7: Main Execution
if __name__ == '__main__':
    try:
        os.makedirs('static', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        df = load_data()
        X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
        model = train_model(X_train, y_train, X_test, y_test)
        create_visualizations(df, model, X_test, y_test, features)
        save_artifacts(model, scaler)
        logger.info("Starting Flask server")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise