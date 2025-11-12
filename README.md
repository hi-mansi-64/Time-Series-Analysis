**📊 Time Series Analysis – Airline Passenger Forecasting
Overview**

This project demonstrates time series analysis and forecasting using the classic Airline Passengers dataset (1949–1960). The goal is to:

Analyze historical trends and seasonal patterns in airline passenger traffic.

Apply ARIMA / SARIMA models for accurate forecasting.

Generate actionable business insights for airline capacity planning.

By following this project, you will learn how to manipulate time series data, visualize trends, apply statistical forecasting models, and evaluate their performance.

**🌟 Dataset**
Feature	Description
Month	Date (Monthly format, Jan 1949 – Dec 1960)
Passengers	Total airline passengers in that month

Total Records: 144 (12 years × 12 months)

Source: Included in this repository as airline-passengers.csv

**Technologies & Libraries Used**

Python 3.11 for pmdarima installation

Pandas & NumPy – Data manipulation & analysis

Matplotlib – Visualization of trends, seasonality, and forecasts

Statsmodels – ARIMA & SARIMA modeling

scikit-learn – Model evaluation (RMSE)

pmdarima – Automated ARIMA parameter selection

**Step-by-Step Implementation**

1️⃣ Load & Explore Data
import pandas as pd
df = pd.read_csv("airline-passengers.csv", parse_dates=['Month'], index_col='Month')
print(df.head())
df.info()

✅ Check for null values, data types, and monthly passenger distribution.

**2️⃣ Trend & Seasonality Decomposition
**
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df, model='multiplicative')
decomposition.plot()


Decomposes series into: Trend, Seasonal, Residual

**Insights:**

Strong upward trend over 12 years

Seasonal peaks every July, troughs in Feb

**Output:**


3️⃣ Moving Average Smoothing
df['MA_6'] = df['Passengers'].rolling(window=6).mean()
df['MA_12'] = df['Passengers'].rolling(window=12).mean()


MA_6: Shows short-term seasonal fluctuations

MA_12: Smooths noise, highlights long-term trend

**📊 Visualization:**


4️⃣ ARIMA / SARIMA Forecasting

Split Data: Last 12 months as test data

Model Selection: Used auto_arima to find best (p,d,q)(P,D,Q)s

Best Model: SARIMA(3,0,0)(0,1,0)[12]

from statsmodels.tsa.arima.model import ARIMA
train = df.iloc[:-12]
test = df.iloc[-12:]
model = ARIMA(train, order=(3,0,0), seasonal_order=(0,1,0,12))
result = model.fit()
forecast = result.forecast(steps=12)


**📈 Forecast vs Actual:**

RMSE: 17.8 passengers (~2.8% error)

** Key Results & Insights**
Aspect	Analysis	Business Insight
Trend	Passenger traffic tripled (1949–1960)	Strong upward demand; plan fleet expansion
Seasonality	July peaks ~30% above average	Increase capacity in Q3
Forecast 1961	550–600 passengers/month	Predictive planning for airline management
Model Performance	RMSE = 17.8	High prediction accuracy

**Skills Demonstrated**
:Data Cleaning & Preprocessing for time series

**Visualization:** Trend, Seasonality, Moving Averages

ARIMA/SARIMA modeling & parameter tuning

Forecast evaluation & RMSE interpretation

Business insight extraction from data

**How to Run**

Clone the repository

git clone https://github.com/hi-mansi-64/Time-Series-Analysis.git
cd Time-Series-Analysis


**Install dependencies**
pip install pandas numpy matplotlib statsmodels scikit-learn pmdarima


Run the analysis

python Time_Series_Analysis.py


**Output Files**

forecast_results.csv → Monthly forecast values

Plots saved automatically (Trend, MA, Forecast)

**🌟 Notes**

Project is beginner-friendly yet professional.

Can be used as reference for forecasting business demand.

Encourages hands-on learning with ARIMA models and Python.
