from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + CSV
model_data = joblib.load("food.pkl")
wastage_model = model_data["models"]["wastage_model"]
label_encoders = model_data["label_encoders"]
features = model_data["features"]

df = pd.read_csv("food_wastage_with_weekend.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["DayOfWeek"] = df["Date"].dt.day_name()

# Add missing derived columns
df["CookingTime"] = pd.to_datetime(df["Time of cooking"], format="%H:%M").dt.hour
df["ExpiryTime"] = pd.to_datetime(df["FoodExpiryTime"], format="%H:%M", errors='coerce').dt.hour
df["ShelfLife"] = (df["ExpiryTime"] - df["CookingTime"]) % 24
df["Wastage_Ratio"] = df["Food_Waste_kg"] / df["FoodPrepared_kg"]
df["Consumption_Ratio"] = df["FoodConsumed_kg"] / df["FoodPrepared_kg"]


# Preprocess matching
def encode(col, val):
    le = label_encoders.get(col)
    return le.transform([val])[0] if le and val in le.classes_ else 0

@app.get("/predict-tomorrow-wastage/")
def predict_tomorrow_wastage():
    tomorrow = datetime.now() + timedelta(days=1)
    day_name = tomorrow.strftime("%A")
    month_name = tomorrow.strftime("%B" \
    "")
    day_of_month = tomorrow.day
    week_of_year = tomorrow.isocalendar()[1]
    season = (tomorrow.month % 12) // 3 + 1
    is_weekend = 1 if tomorrow.weekday() >= 5 else 0

    df_day = df[df["DayOfWeek"] == day_name]
    meals = ["Breakfast", "Lunch", "Dinner"]
    predictions = []

    for meal in meals:
        df_meal = df_day[df_day["MealType"] == meal]
        if df_meal.empty:
            continue

        # Mean values for this day+meal combo
        mean_row = df_meal.mean(numeric_only=True)

        input_dict = {
            "Month": encode("Month", month_name),
            "DayOfWeek": encode("DayOfWeek", day_name),
            "DayOfMonth": day_of_month,
            "WeekOfYear": week_of_year,
            "Season": season,
            "MealType": encode("MealType", meal),
            "ExpectedFootfall": 300,
            "Weather": encode("Weather", df_meal["Weather"].mode().iloc[0]),
            "weekend": is_weekend,
            "Storage status": encode("Storage status", df_meal["Storage status"].mode().iloc[0]),
            "Stored in fridge / open tray": encode("Stored in fridge / open tray", df_meal["Stored in fridge / open tray"].mode().iloc[0]),
            "Footfall_Difference": mean_row["ExpectedFootfall"] - mean_row["ActualFootfall"],
            "CookingTime": mean_row["CookingTime"],
            "ShelfLife": mean_row["ShelfLife"],
            "Wastage_Ratio_7d_MA": mean_row["Wastage_Ratio"],
            "Wastage_Ratio_14d_MA": mean_row["Wastage_Ratio"],
            "Consumption_Ratio_7d_MA": mean_row["Consumption_Ratio"],
            "Consumption_Ratio_14d_MA": mean_row["Consumption_Ratio"]
        }

        input_df = pd.DataFrame([input_dict])[features]
        pred = wastage_model.predict(input_df)[0]

        predictions.append({
            "meal": meal,
            "predicted_waste_kg": round(pred, 2)
        })
    print(predictions)

    return {
        "tomorrow": tomorrow.strftime("%A, %d-%m-%Y"),
        "predictions": predictions
    }

from fastapi import Request
from pydantic import BaseModel
import subprocess

class FoodEntry(BaseModel):
    date: str
    food_item: str
    quantity_prepared: float
    quantity_consumed: float
    meal_type: str

@app.post("/submit-food-log/")
def submit_food_log(entry: FoodEntry):
    # Derive other fields
    date = pd.to_datetime(entry.date, dayfirst=False)
    month = date.strftime('%B')
    day_of_week = date.strftime('%A')
    weekend = 'yes' if date.weekday() >= 5 else 'no'
    
    cooked_time = {
        'Breakfast': '07:30',
        'Lunch': '13:00',
        'Dinner': '20:00'
    }[entry.meal_type]
    expiry_time = {
        'Breakfast': '11:30',
        'Lunch': '17:00',
        'Dinner': '00:00'
    }[entry.meal_type]
    
    actual_footfall = 280  # You can improve this later with actual data
    expected_footfall = 300
    waste_kg = round(entry.quantity_prepared - entry.quantity_consumed, 2)

    new_row = {
        "Date": date.strftime('%d-%m-%Y'),
        "Month": month,
        "DayOfWeek": day_of_week,
        "MealType": entry.meal_type,
        "MainCourse": entry.food_item,
        "Curry": "Rajma",
        "SideDish": "Pickle",
        "ExpectedFootfall": expected_footfall,
        "ActualFootfall": actual_footfall,
        "FoodPrepared_kg": entry.quantity_prepared,
        "FoodConsumed_kg": entry.quantity_consumed,
        "Food_Waste_kg": waste_kg,
        "Time of cooking": cooked_time,
        "FoodExpiryTime": expiry_time,
        "Storage status": "Leftover",
        "Stored in fridge / open tray": "Open tray",
        "Weather": "Pleasant",
        "weekend": weekend
    }

    df = pd.read_csv("food_wastage_with_weekend.csv")
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("food_wastage_with_weekend.csv", index=False)

    # Re-train model by invoking generate.py
    subprocess.run(["python", "generate.py"])

    return {"message": "Data submitted and model retrained successfully"}