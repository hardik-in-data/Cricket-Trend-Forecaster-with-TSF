# Time Series Tuned Cricket Insight Forecaster

## Overview

This project leverages advanced time series analysis to forecast professional cricket player performances. It offers detailed visualizations and predictive insights across batting and bowling metrics, with an emphasis on statistical precision, model tuning, and forecast accuracy.

## Key Features

- **Comprehensive Player Analysis** - Deep dive into player performance metrics
- **Multi-dimensional Visualization** - Performance breakdowns by opposition teams and venues
- **Time Series Forecasting** - Predict future performance using ARIMA, SARIMA, and baseline models
- **Comparative Model Evaluation** - Automatic selection of best forecasting model based on MSE/MAE metrics
- **Statistical Validation** - Includes stationarity testing, ACF/PACF analysis, and proper model diagnostics
- **Automated Parameter Selection** - Grid search and auto_arima for optimal model parameters

## Usage

### Primary Workflow

```python
# For bowler analysis
# 1. Load and prepare player data
bowler_df = filtered_df_bowling[filtered_df_bowling["player_name"] == "Pat Cummins"]
bowler_df.set_index("Match Start Date", inplace=True)

# 2. Extract time series data
time_series_bowler_data = bowler_df[['wickets', 'economy', 'bowling_average', 'strike_rate']]

# 3. Split into training and test sets
train_size = int(len(time_series_bowler_data) * 0.8)
train_data = time_series_bowler_data[:train_size]
test_data = time_series_bowler_data[train_size:]

# 4. Initialize and run forecaster
bowling_forecaster = TimeSeriesForecaster(train_data, test_data)
bowling_forecaster.fit_models()
bowling_forecaster.visualize()
bowling_future, bowling_naive_future = bowling_forecaster.forecast_future(steps=5)
```

### Similar workflow for batting analysis:

```python
# For batsman analysis
batter_df = filtered_df_batting[filtered_df_batting["player_name"] == "Travis Head"]
batter_df.set_index("Match Start Date", inplace=True)
time_series_batter_data = batter_df[['runs', 'batting_average', 'strikeRate']]

# Split, initialize and forecast
train_size = int(len(time_series_batter_data) * 0.8)
train_data = time_series_batter_data[:train_size]
test_data = time_series_batter_data[train_size:]

batting_forecaster = TimeSeriesForecaster(train_data, test_data)
batting_forecaster.fit_models()
batting_forecaster.visualize()
batting_future, batting_naive_future = batting_forecaster.forecast_future(steps=5)
```

## Data Sources

The project uses the following datasets:
- `test_Bowling_Card.csv` - Detailed bowling statistics from test matches
- `test_Batting_Card.csv` - Detailed batting statistics from test matches
- `players_info.csv` - Player biographical information
- `test_Matches_Data.csv` - Match information and results

## Workflow & Implementation

The project follows a structured data science workflow:

1. **Data Loading & Inspection**
   - Load cricket match data from multiple CSV files
   - Explore basic statistics and check for missing values
   - Identify potential issues in the datasets

2. **Data Cleaning & Preprocessing**
   - Handle missing values with appropriate strategies (zeros for numeric stats, "Unknown" for categorical)
   - Drop unnecessary columns with excessive missing data
   - Filter relevant player data

3. **Feature Engineering**
   - Calculate bowling metrics: bowling average, strike rate, economy
   - Calculate batting metrics: batting average, strike rate
   - Aggregate performance by match, opposition, and venue

4. **Performance Analysis**
   - Group and analyze performance against different opposition teams
   - Analyze performance at different venues
   - Track performance metrics over time

5. **Time Series Analysis**
   - Test for stationarity using Augmented Dickey-Fuller test
   - Generate ACF and PACF plots to identify potential model parameters
   - Split data into training and testing sets (80/20 split)

6. **Model Development & Comparison**
   - Implement multiple forecasting approaches (ARIMA, SARIMA, Naive baseline)
   - Perform grid search and auto_arima for optimal parameter selection
   - Evaluate models using MSE and MAE metrics

7. **Future Forecasting**
   - Generate forecasts for future matches using the best-performing model
   - Compare with naive forecasting approach
   - Provide performance predictions for upcoming matches

## Technical Analysis Methods

### Player Performance Analysis
- **Opposition Analysis**: Aggregation and comparison of performance metrics against different teams
- **Venue Analysis**: Statistical breakdown of performance at different cricket grounds
- **Metric Calculation**:
  - Bowling: Wickets, Economy, Bowling Average (runs/wickets), Strike Rate (balls/wicket)
  - Batting: Runs, Batting Average (runs/dismissals), Strike Rate (runs*100/balls)

### Statistical Analysis
1. **Stationarity Testing**:
   ```python
   from statsmodels.tsa.stattools import adfuller
   
   def adf_test(series, metric_name):
       series_clean = pd.Series(series).dropna()
       result = adfuller(series_clean)
       is_stationary = result[1] <= 0.05
       return result[1], is_stationary
   ```

2. **Autocorrelation Analysis**:
   ```python
   def plot_acf_pacf(train_data, lags=20):
       for column in train_data.columns:
           series = train_data[column].dropna()
           allowed_lags = min(lags, max(1, (len(series) // 2) - 1))
           
           # ACF & PACF plots
           sm.graphics.tsa.plot_acf(series, lags=allowed_lags)
           sm.graphics.tsa.plot_pacf(series, lags=allowed_lags)
   ```

## Forecasting Methodology

The core of this project is the `TimeSeriesForecaster` class that implements multiple forecasting approaches:

```python
class TimeSeriesForecaster:
    def __init__(self, train_data, test_data, seasonal_period=12):
        # Initialize with train-test data
        self.train_data = train_data
        self.test_data = test_data
        self.seasonal_period = seasonal_period
        
        # Storage for models and forecasts
        self.arima_models = {}
        self.sarima_models = {}
        self.best_models = {}
        self.forecasts = {}
        self.naive_forecasts = {}
        self.metrics = {}
```

### Key Methods:

1. **Grid Search for ARIMA Parameters**:
   ```python
   def grid_search_arima(self, train_data, test_data):
       best_score, best_order = float("inf"), None
       for p, d, q in product(self.p_values, self.d_values, self.q_values):
           try:
               model = ARIMA(train_data, order=(p, d, q))
               fitted_model = model.fit()
               forecast = fitted_model.forecast(steps=len(test_data))
               mse = mean_squared_error(test_data, forecast)
               if mse < best_score:
                   best_score, best_order = mse, (p, d, q)
           except Exception as e:
               continue
       return best_order
   ```

2. **ARIMA Model Fitting**:
   ```python
   def fit_arima(self, train_data, test_data):
       best_arima_order = self.grid_search_arima(train_data, test_data)
       if best_arima_order is None:
           # Fallback to auto_arima
           auto_arima_model = auto_arima(train_data, ...)
           best_arima_order = auto_arima_model.order
           
       arima_model = ARIMA(train_data, order=best_arima_order)
       fitted_arima = arima_model.fit()
       arima_forecast = fitted_arima.forecast(steps=len(test_data))
       return fitted_arima, arima_forecast, best_arima_order
   ```

3. **SARIMA Model Fitting**:
   ```python
   def fit_sarima(self, train_data, test_data):
       auto_sarima = auto_arima(
           train_data, seasonal=True, m=self.seasonal_period, ...
       )
       
       best_sarima_order = auto_sarima.order
       best_sarima_seasonal_order = auto_sarima.seasonal_order
       sarima_model = SARIMAX(train_data, order=best_sarima_order, 
                             seasonal_order=best_sarima_seasonal_order)
       fitted_sarima = sarima_model.fit(disp=False)
       sarima_forecast = fitted_sarima.forecast(steps=len(test_data))
       return fitted_sarima, sarima_forecast, best_sarima_order, best_sarima_seasonal_order
   ```

4. **Model Selection and Forecasting**:
   ```python
   def fit_models(self):
       for column in self.train_data.columns:
           # Fit ARIMA and SARIMA models
           fitted_arima, arima_forecast, best_arima_order = self.fit_arima(...)
           fitted_sarima, sarima_forecast, best_sarima_order, best_sarima_seasonal_order = self.fit_sarima(...)
           
           # Generate naive forecast
           naive_forecast = np.full(self.n_test, self.train_data[column].iloc[-1])
           
           # Calculate error metrics and select best model
           arima_mse = mean_squared_error(self.test_data[column], arima_forecast)
           sarima_mse = mean_squared_error(self.test_data[column], sarima_forecast)
           naive_mse = mean_squared_error(self.test_data[column], naive_forecast)
           
           # Select best model based on lowest MSE
           if arima_mse < sarima_mse and arima_mse < naive_mse:
               self.best_models[column] = {'model': 'ARIMA', ...}
           elif sarima_mse < arima_mse and sarima_mse < naive_mse:
               self.best_models[column] = {'model': 'SARIMA', ...}
           else:
               self.best_models[column] = {'model': 'Naive', ...}
   ```

5. **Future Forecasting**:
   ```python
   def forecast_future(self, steps=5):
       future_forecasts = {}
       future_naive_forecasts = {}
       
       for column in self.train_data.columns:
           if self.best_models[column]['model'] in ['ARIMA', 'SARIMA']:
               future_forecast = self.best_models[column]['fitted_model'].forecast(steps=steps)
           else:
               future_forecast = np.full(steps, self.train_data[column].iloc[-1])
           future_forecasts[column] = future_forecast
           
           # Also generate naive forecasts for comparison
           last_value = self.train_data[column].iloc[-1]
           future_naive = np.full(steps, last_value)
           future_naive_forecasts[column] = future_naive
           
       return future_forecasts, future_naive_forecasts
   ```

## Results & Evaluation

The project evaluates forecasting models using:

1. **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values
2. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values

For each performance metric (wickets, economy, bowling average, etc.), the system:

1. Compares ARIMA, SARIMA, and Naive forecasting models
2. Automatically selects the best performing model based on lowest test MSE
3. Provides detailed error metrics for model comparison
4. Generates visualizations of actual vs. predicted values
5. Uses the best model to forecast future performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

