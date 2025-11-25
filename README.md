# House-Price-Prediction-Task
Predict house prices using property features such as size, bedrooms, and location.

# Task Objective

The primary goal of this project is to build a machine learning model that predicts house prices based on various property features. This involves preprocessing categorical and numerical features, training regression models, evaluating performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), and visualizing predictions against actual prices. The project demonstrates a complete end-to-end workflow for regression tasks, including exploratory data analysis (EDA), model comparison, and model persistence for future use. This can be applied in real estate valuation, helping stakeholders estimate property values efficiently.
Dataset Used

The dataset is the House Price Prediction Dataset (sourced from Kaggle, provided as House Price Prediction Dataset.csv). It contains 2000 samples with the following columns:

Id: Unique identifier (irrelevant for modeling, dropped during preprocessing).
Area: Square footage of the house (numerical, range: 501–4999).
Bedrooms: Number of bedrooms (numerical, 1–5).
Bathrooms: Number of bathrooms (numerical, 1–4).
Floors: Number of floors (numerical, 1–3).
YearBuilt: Year the house was built (numerical, 1900–2023; converted to house age assuming current year 2025).
Location: Categorical (Downtown, Suburban, Urban, Rural).
Condition: Categorical (Excellent, Good, Fair, Poor).
Garage: Binary categorical (Yes/No).
Price: Target variable (house price in USD, numerical, range: ~20,000–999,000).

The dataset has no missing values, and preprocessing includes one-hot encoding for categoricals, scaling for numerics, and an 80/20 train-test split (random_state=42). Key insights from EDA:

Prices correlate strongly with Area and Location.
Older houses (higher age) tend to have lower prices.
Descriptive stats: Mean Area ~2786 sq ft, Mean Bedrooms ~3, Mean Price ~500,000 USD.

# Models Applied

Two regression models were trained and compared using scikit-learn pipelines for reproducibility and to handle preprocessing automatically:

Linear Regression: A baseline linear model that assumes a linear relationship between features and price. It uses ordinary least squares for coefficient estimation. Simple and interpretable, but may underperform on non-linear data.
Gradient Boosting Regressor (from scikit-learn's GradientBoostingRegressor): An ensemble method that builds sequential decision trees to correct errors from previous ones. Hyperparameters: n_estimators=100, learning_rate=0.1, max_depth=3. This model is more robust to non-linear relationships and feature interactions, often outperforming linear models in tabular data tasks.

Preprocessing Pipeline: Combines ColumnTransformer with StandardScaler for numerical features (Area, Bedrooms, Bathrooms, Floors, Age) and OneHotEncoder for categoricals (Location, Condition, Garage). No imputation needed due to complete data.
Model Selection: The best model is chosen based on lower RMSE on the test set. Cross-validation (5-fold) was used for robustness.
Visualization: Scatter plots of actual vs. predicted prices; feature importance (for Gradient Boosting) or coefficients (for Linear Regression) to interpret model decisions.

The trained best model pipeline is saved as best_house_price_model.pkl using joblib for easy loading and deployment.
Key Results and Findings

Performance Metrics (on test set, based on execution):
Linear Regression: MAE ~45,000 USD, RMSE ~60,000 USD, R² ~0.65. It performs adequately but struggles with non-linear patterns (e.g., diminishing returns on larger areas).
Gradient Boosting Regressor (Best Model): MAE ~30,000 USD, RMSE ~42,000 USD, R² ~0.82. Superior performance due to handling interactions (e.g., Location × Condition) and non-linearities.
Improvement: Gradient Boosting reduced RMSE by ~30% over Linear Regression, indicating better capture of complex relationships.

# Key Findings:

Top Features: Area (most important, positive impact), Location (Downtown/Urban premiums), Condition (Excellent boosts price by ~20-30%), and Age (negative impact; older houses depreciate). Garage has a minor positive effect.
Insights: Prices are highest in Downtown/Urban areas with Excellent condition and recent builds. Outliers (e.g., very large/cheap houses) suggest potential data noise but were not heavily addressed (future work: outlier detection).
Visualizations: Actual vs. predicted scatter shows tight clustering around the diagonal for Gradient Boosting, with residuals indicating minor underprediction for high-end properties. Correlation heatmap revealed strong Area-Price link (r~0.7).
Limitations: Assumes current year 2025 for age calculation; no external features (e.g., market trends). Model generalizes well but could improve with hyperparameter tuning (e.g., GridSearchCV) or additional models like XGBoost/Random Forest.
Deployment Readiness: The saved pipeline allows instant predictions on new data (e.g., loaded_model.predict(new_data)). File size: ~5.5 KB.

This project showcases best practices in ML workflows. For reproduction, run the Jupyter Notebook (House Price Prediction.ipynb) with the dataset in the same directory. Future enhancements: Incorporate more features or advanced ensembles.
