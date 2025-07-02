import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load the dataset
df = pd.read_csv('food_wastage_with_weekend.csv')
df['weekend'] = df['weekend'].map({'yes': 1, 'no': 0})

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['DayOfWeek'] = df['Date'].dt.day_name()
df['DayOfMonth'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
df['Season'] = df['Date'].dt.month % 12 // 3 + 1
df['CookingTime'] = pd.to_datetime(df['Time of cooking'], format='%H:%M').dt.hour
df['ExpiryTime'] = pd.to_datetime(df['FoodExpiryTime'], format='%H:%M').dt.hour
df['ShelfLife'] = (df['ExpiryTime'] - df['CookingTime']) % 24
df['Wastage_Ratio'] = df['Food_Waste_kg'] / df['FoodPrepared_kg']
df['Footfall_Difference'] = df['ExpectedFootfall'] - df['ActualFootfall']
df['Consumption_Ratio'] = df['FoodConsumed_kg'] / df['FoodPrepared_kg']

df = df.sort_values('Date')
for window in [7, 14]:
    df[f'Wastage_Ratio_{window}d_MA'] = df.groupby(['MealType', 'DayOfWeek'])['Wastage_Ratio'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df[f'Consumption_Ratio_{window}d_MA'] = df.groupby(['MealType', 'DayOfWeek'])['Consumption_Ratio'].transform(lambda x: x.rolling(window, min_periods=1).mean())

df['MainCourse_Curry'] = df['MainCourse'] + '_' + df['Curry']
df['Meal_Combo'] = df['MainCourse'] + df['Curry'] + df['SideDish']

# Encode categorical features
label_encoders = {}
categorical_cols = ['Month', 'DayOfWeek', 'MealType', 'MainCourse', 'Curry', 'SideDish',
                    'Storage status', 'Stored in fridge / open tray', 'Weather', 'Season',
                    'MainCourse_Curry', 'Meal_Combo']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Select features and targets
features = ['Month', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Season',
            'MealType', 'ExpectedFootfall', 'Weather', 'weekend',
            'Storage status', 'Stored in fridge / open tray',
            'Footfall_Difference', 'CookingTime', 'ShelfLife',
            'Wastage_Ratio_7d_MA', 'Wastage_Ratio_14d_MA',
            'Consumption_Ratio_7d_MA', 'Consumption_Ratio_14d_MA']

targets = ['Food_Waste_kg', 'MainCourse', 'Curry', 'SideDish', 'CookingTime', 'ShelfLife']

X = df[features]
y = df[targets]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'wastage_model': RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=5),
    'main_course_model': RandomForestClassifier(n_estimators=200, random_state=42),
    'curry_model': RandomForestClassifier(n_estimators=200, random_state=42),
    'side_dish_model': RandomForestClassifier(n_estimators=200, random_state=42),
    'cooking_time_model': RandomForestRegressor(n_estimators=100, random_state=42),
    'shelf_life_model': RandomForestRegressor(n_estimators=100, random_state=42)
}

models['wastage_model'].fit(X_train, y_train['Food_Waste_kg'])
models['main_course_model'].fit(X_train, y_train['MainCourse'])
models['curry_model'].fit(X_train, y_train['Curry'])
models['side_dish_model'].fit(X_train, y_train['SideDish'])
models['cooking_time_model'].fit(X_train, y_train['CookingTime'])
models['shelf_life_model'].fit(X_train, y_train['ShelfLife'])

# Save everything to a single pickle file
package = {
    "models": models,
    "label_encoders": label_encoders,
    "features": features
}

joblib.dump(package, 'food.pkl')
print("✅ Pickle file 'food.pkl' generated successfully.")