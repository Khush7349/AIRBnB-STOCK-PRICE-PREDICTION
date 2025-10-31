# AIRBnB-STOCK-PRICE-PREDICTION
An end-to-end time-series project predicting ABNB stock prices. This analysis demonstrates a correct forecasting methodology (no data leakage) and finds that a simple Naive Forecast (Random Walk Hypothesis) outperforms traditional machine learning models.

## Project Goal
The primary goal of this project is to build and evaluate a machine learning model to forecast the next day's closing price of Airbnb (ABNB) stock using its historical data.

This project demonstrates the critical importance of a correct time-series methodology. It specifically addresses common pitfalls like **data leakage** (using future information to predict the present) and **improper validation** (using a random shuffle instead of a chronological split).

## 📊 Dataset
The data used is the "Airbnb Inc (ABNB) Stock Market Analysis" dataset, publicly available on Kaggle.

* **Source:** [Airbnb Inc (ABNB) Stock Market Analysis on Kaggle](https://www.kaggle.com/datasets/whenamancodes/airbnb-inc-stock-market-analysis)
* **File Used:** `ABNB.csv`

---

## ⚙️ Methodology

The notebook follows a rigorous, sequential process to ensure the forecast is realistic and valid.

### 1. Data Loading & EDA
* Loaded the `ABNB.csv` data into a pandas DataFrame.
* Converted the `Date` column to a datetime object and set it as the index to ensure chronological order.
* Visualized the `Close` price and `Volume` over time to identify trends and patterns.

### 2. Feature Engineering (No Data Leakage)
To prevent the model from "cheating" by seeing same-day data, all features were created using **past (lagged) data**. This simulates a real-world scenario where you are predicting tomorrow's price using only information you have today.

Engineered features included:
* `lag_1`: The closing price from the previous day.
* `lag_7`: The closing price from 7 days ago.
* `MA_7`: A 7-day rolling moving average of the *lagged* close price.
* `MA_30`: A 30-day rolling moving average.
* `Volume_lag_1`: The trading volume from the previous day.

### 3. Chronological Train-Test Split
The data was split **chronologically** to respect the time-series nature of the data.
* **Training Set:** The first 80% of the data.
* **Test Set:** The final 20% of the data.

This method ensures the model is trained only on the past and validated on the unseen future.

### 4. Modeling & Evaluation
Four different models were built and evaluated on the test set using R-squared (R2) and Root Mean Squared Error (RMSE):
1.  **Decision Tree Regressor**
2.  **Random Forest Regressor**
3.  **XGBoost Regressor**
4.  **Naive Forecast (Baseline)**: A simple model where the prediction for tomorrow is just the price from today (`lag_1`).

---

## 🚀 Key Findings & Conclusion

The key finding of this project is that the "smart" machine learning models were all outperformed by the simple baseline, which supports the **"Random Walk Hypothesis"** for this stock.

### Model Comparison (on Test Set)
* **Naive Forecast (Baseline):**
    * **RMSE: 3.785** (Lowest Error)
    * **R2 Score: 0.860** (Best Fit)

* **Decision Tree:**
    * RMSE: 10.303
    * R2 Score: -0.037

* **Random Forest:**
    * RMSE: 11.013
    * R2 Score: -0.185

* **XGBoost:**
    * RMSE: 13.728
    * R2 Score: -0.841

### Conclusion
The machine learning models (Decision Tree, Random Forest, XGBoost) all performed very poorly, resulting in **negative R-squared scores**. A negative R2 indicates that the model's predictions are *worse* than simply guessing the average price of the test set.

The **Naive Forecast** was the clear winner, with the lowest error (RMSE: 3.785) and the only positive, strong R2 score (0.860). This demonstrates that for this dataset, the best and most reliable predictor for tomorrow's price is simply today's price. The additional features based on past trends (like moving averages) only added noise and confused the complex models, leading to worse performance.

---

## 🛠️ Libraries Used
* `pandas`
* `numpy`
* `seaborn`
* `matplotlib`
* `scikit-learn` (DecisionTreeRegressor, RandomForestRegressor, metrics)
* `xgboost` (XGBRegressor)
