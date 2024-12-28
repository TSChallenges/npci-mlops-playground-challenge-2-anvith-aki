import gradio as gr
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("random_forest_model.pkl")

# Prediction function
def predict_bike_rentals(season, hour, holiday, working_day, weather, temp, atemp, humidity, windspeed, year, month, weekday):
    # Map categorical inputs to numerical values
    season_mapping = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
    weather_mapping = {"Clear": 1, "Mist": 2, "Light Rain": 3, "Heavy Rain": 4}
    weekday_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    
    # One-hot encode the selected weekday
    weekday_encoding = [0] * 7
    weekday_encoding[weekday_mapping[weekday]] = 1

    # Combine all features into a single array
    features = np.array([season_mapping[season], hour, holiday, working_day, weather_mapping[weather], temp, atemp, humidity, windspeed, year, month] + weekday_encoding)
    
    # Predict using the model
    prediction = model.predict(features.reshape(1, -1))
    return int(prediction[0])

# Gradio Interface
inputs = [
    gr.Dropdown(choices=["Spring", "Summer", "Fall", "Winter"], label="Season"),
    gr.Slider(minimum=0, maximum=23, label="Hour of Day"),
    gr.Checkbox(label="Holiday"),
    gr.Checkbox(label="Working Day"),
    gr.Dropdown(choices=["Clear", "Mist", "Light Rain", "Heavy Rain"], label="Weather Condition"),
    gr.Slider(minimum=-10, maximum=40, label="Temperature (°C)"),
    gr.Slider(minimum=-10, maximum=40, label="Feels Like Temperature (°C)"),
    gr.Slider(minimum=0, maximum=100, label="Humidity (%)"),
    gr.Slider(minimum=0, maximum=50, label="Wind Speed (km/h)"),
    gr.Dropdown(choices=[0, 1], label="Year (0: 2011, 1: 2012)"),
    gr.Slider(minimum=1, maximum=12, label="Month"),
    gr.Dropdown(choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], label="Weekday")
]

outputs = gr.Textbox(label="Predicted Bike Rentals")

# Launch Gradio Interface
app = gr.Interface(fn=predict_bike_rentals, inputs=inputs, outputs=outputs, title="Bike Rental Count Predictor")
app.launch(server_name = "0.0.0.0", server_port = 7860)
