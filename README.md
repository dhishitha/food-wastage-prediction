# 🍲 FoodShare AI – Smart Food Redistribution Platform

This is an AI-powered web platform that predicts food surplus in large-scale kitchens (like cafeterias) and facilitates donation to NGOs for reducing food wastage.

---

## 🛠️ Features

- 📊 Predicts food surplus for the next day using a machine learning model (Random Forest).
- 🧠 Retrains models dynamically with new data submissions.
- 🏠 Interactive frontend with separate dashboards for kitchens and NGOs.
- 🔁 Real-time donation coordination between kitchens and NGOs.

---

## 📁 Project Structure

├──generate.py # Trains and serializes ML models
├── main.py # FastAPI backend with prediction and submission endpoints
├── ui.html # Frontend dashboard (Kitchen + NGO)
├── food_wastage_with_weekend.csv # Dataset (should be in project root)
├── food.pkl # Serialized model and encoders (auto-generated)
