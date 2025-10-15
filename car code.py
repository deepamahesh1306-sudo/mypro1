# ============================================================
# 1?? Import Required Libraries
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 2?? Load Dataset
# ============================================================
# Replace with your dataset file name or path
df = pd.read_csv("car_data.csv")

print("? Dataset Loaded Successfully")
print(df.head())
print(df.info())

# ============================================================
# 3?? Data Cleaning
# ============================================================
print("\n?? Checking Missing Values:\n", df.isnull().sum())

# Fill numeric missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Remove units (example: '20 km/l' ? 20)
def remove_units(x):
    if isinstance(x, str):
        return float(x.split()[0])
    return x

for col in ['Mileage', 'Engine']:
    if col in df.columns:
        df[col] = df[col].apply(remove_units)

# ============================================================
# 4?? Outlier Detection & Removal
# ============================================================
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ['Selling_Price', 'KM_Driven']:
    if col in df.columns:
        df = remove_outliers(df, col)

print("\n? After outlier removal, dataset shape:", df.shape)

# ============================================================
# 5?? Feature Engineering
# ============================================================
# Create Car Age if Year column exists
if 'Year' in df.columns:
    df['Car_Age'] = 2025 - df['Year']

# Price per KM (avoid divide by zero)
if 'Selling_Price' in df.columns and 'KM_Driven' in df.columns:
    df['Price_per_KM'] = df['Selling_Price'] / (df['KM_Driven'] + 1)

# Age Category
def age_category(age):
    if age <= 3:
        return 'New'
    elif age <= 8:
        return 'Mid'
    else:
        return 'Old'

if 'Car_Age' in df.columns:
    df['Age_Category'] = df['Car_Age'].apply(age_category)

print("\n? Feature Engineering Done!")

# ============================================================
# 6?? Encoding Categorical Variables
# ============================================================
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\n?? Encoded Columns:", list(cat_cols))

# ============================================================
# 7?? Feature Scaling
# ============================================================
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

df[num_cols] = scaler.fit_transform(df[num_cols])

print("\n? Feature Scaling Completed")

# ============================================================
# 8?? Split Data into Train/Test
# ============================================================
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n? Train/Test Split Done")
print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# ============================================================
# 9?? Model Building — Linear Regression
# ============================================================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

print("\n?? Linear Regression Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R²:", r2_score(y_test, y_pred_lr))

# ============================================================
# ?? Model Building — Random Forest
# ============================================================
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n?? Random Forest Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R²:", r2_score(y_test, y_pred_rf))

# ============================================================
# 11?? Hyperparameter Tuning (GridSearchCV)
# ============================================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

print("\n? Best Random Forest Parameters:", grid.best_params_)
best_rf = grid.best_estimator_

# ============================================================
# 12?? Evaluation of Tuned Model
# ============================================================
y_pred_best = best_rf.predict(X_test)

print("\n?? Tuned Random Forest Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred_best))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best)))
print("R²:", r2_score(y_test, y_pred_best))

# ============================================================
# 13?? Feature Importance
# ============================================================
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance - Random Forest")
plt.show()

