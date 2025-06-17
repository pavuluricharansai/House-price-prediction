# California Housing Price Prediction

This project uses machine learning to predict house prices in California based on various features such as median income, house age, average number of rooms, and more. The project leverages the Random Forest Regressor model to make predictions, and the Flask framework to create an API for serving the model. Additionally, the project includes data preprocessing, model training, evaluation, and visualization components.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

The project performs the following tasks:

1. **Data Collection**: Loads the California housing dataset.
2. **Data Preprocessing**: Cleans the dataset and splits it into training and testing sets, scaling the features.
3. **Model Training**: Trains a Random Forest Regressor model to predict house prices.
4. **Visualization**: Generates feature importance and actual vs. predicted price plots.
5. **Model Serving**: The trained model is saved, and a Flask web app is set up to serve predictions based on user input.

## Installation

Follow the steps below to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/california-housing-price-prediction.git
cd california-housing-price-prediction
