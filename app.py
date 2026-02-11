import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("weather.csv")

# Create Rain column
df['Rain'] = df['Weather'].apply(lambda x: 1 if 'Rain' in x else 0)

# Features and Target
X = df[['Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Press_kPa']]
y = df['Rain']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌦️ Weather Rain Prediction App")

st.write("Enter weather details to predict rain.")

temp = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
wind = st.number_input("Wind Speed (km/h)")
pressure = st.number_input("Pressure (kPa)")

if st.button("Predict"):

    input_data = [[temp, humidity, wind, pressure]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🌧️ It will Rain!")
    else:
        st.success("☀️ No Rain Expected!")
