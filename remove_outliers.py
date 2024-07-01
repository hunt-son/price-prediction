import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Load the updated data file
file_path_updated = '../input/normalized_clean_newconstruction_nearwater_tricounty.csv'
df = pd.read_csv(file_path_updated)

# Rename columns to remove spaces
df.columns = [col.replace(' ', '_') for col in df.columns]

# Ensure all data are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# One-hot encode the 'Area' column
df = pd.get_dummies(df, columns=['Area'], drop_first=True)

# Check for missing values and handle them
df = df.dropna()

# Combine #Beds and #FBaths into a new feature Total_Rooms
df['Total_Rooms'] = df['#Beds'] + df['#FBaths']
df.drop(columns=['#Beds', '#FBaths'], inplace=True)

# Log transform skewed variables
df['Log_SqFt_LA'] = np.log1p(df['SqFt_LA'])

# Calculate z-scores to identify outliers
df['z_score'] = zscore(df['Sale_Price'])

# Remove outliers with z-score greater than 3 or less than -3
df_no_outliers = df[(df['z_score'] < 3) & (df['z_score'] > -3)]
df_no_outliers = df_no_outliers.drop(columns=['z_score'])

# Re-define the independent variables (predictors) and dependent variable (response)
X_no_outliers = df_no_outliers.drop(columns=['Sale_Price', 'SqFt_LA'])
y_no_outliers = df_no_outliers['Sale_Price']

# Convert boolean columns to integers
X_no_outliers = X_no_outliers.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# Ensure all data are numeric
X_no_outliers = X_no_outliers.apply(pd.to_numeric, errors='coerce')
y_no_outliers = y_no_outliers.apply(pd.to_numeric, errors='coerce')

# Check the data types
print(X_no_outliers.dtypes)
print(y_no_outliers.dtypes)

# Print the first few rows of X_no_outliers and y_no_outliers to inspect the data
print(X_no_outliers.head())
print(y_no_outliers.head())

# Standardize the features
scaler = StandardScaler()
X_scaled_no_outliers = scaler.fit_transform(X_no_outliers)

# Fit OLS model without outliers
X_with_features_no_outliers = sm.add_constant(X_no_outliers)
ols_with_features_no_outliers = sm.OLS(y_no_outliers, X_with_features_no_outliers).fit()

# Print the summary of the improved OLS model without outliers
print(ols_with_features_no_outliers.summary())

# Residual Analysis
'''
plt.figure(figsize=(10, 6))
plt.scatter(ols_with_features_no_outliers.fittedvalues, ols_with_features_no_outliers.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual Plot Without Outliers')
plt.show()
'''

# Print the summary of the improved OLS model without outliers
ols_summary_no_outliers = ols_with_features_no_outliers.summary()

print(ols_summary_no_outliers)

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Given new data values
total_rooms = 8
sqft_la = 3100
area = 3221.0
waterfront = 0
pool = 1

# Create a new DataFrame with the input data
new_data = pd.DataFrame({
    'Total_Rooms': [total_rooms],
    'SqFt_LA': [sqft_la / 21406],  # Normalize sqft_la
    'Pool_YN': [pool],
    'Waterfront_Property_(Y/N)': [waterfront],
})

# Add area columns from the previous model to new data
area_columns = [col for col in X_with_features_no_outliers.columns if col.startswith('Area_')]

for col in area_columns:
    new_data[col] = 0
    if float(col.split('_')[1]) == area:
        new_data[col] = 1

# Ensure all required columns are present in the new data
for col in X_with_features_no_outliers.columns.drop('const'):
    if col not in new_data.columns:
        new_data[col] = 0

# Ensure the order of columns matches the model's training data
new_data = new_data[X_with_features_no_outliers.columns.drop('const')]

# Add constant term
new_data['const'] = X_with_features_no_outliers['const'].mean()

# Print model coefficients to inspect
print(ols_with_features_no_outliers.params)

# Ensure new data is scaled consistently for Gradient Boosting
scaler = StandardScaler()
scaler.fit(X_no_outliers)  # Fit scaler on the original training data
new_data_scaled = scaler.transform(new_data.drop(columns='const'))

# Fit Gradient Boosting model on the cleaned and scaled data
gbr = GradientBoostingRegressor()
gbr.fit(X_scaled_no_outliers, y_no_outliers)

# Predict using Gradient Boosting model
gbr_predicted_price = gbr.predict(new_data_scaled)
print("Predicted Sale Price (Gradient Boosting):", gbr_predicted_price)