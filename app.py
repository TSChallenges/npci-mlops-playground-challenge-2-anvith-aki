import gradio as gr
import joblib
import numpy as np

# Load the pre-trained model (Ensure you have a model trained and saved)
model = joblib.load("random_forest_model.pkl")

# Prediction function
def predict_bike_rentals(season, hour, holiday, working_day, weather, temp, atemp, humidity, windspeed, year, month, weekday):
    # One-hot encode the selected weekday
    weekday_encoding = [0] * 7  # [0,0,0,0,0,0,0]
    weekday_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    weekday_encoding[weekday_mapping[weekday]] = 1

    # Combine all features into a single array
    features = np.array([
        season, hour, holiday, working_day, weather, temp, atemp, humidity, 
        windspeed, year, month] + weekday_encoding)
    
    # Predict using the model
    prediction = model.predict(features.reshape(1, -1))
    return int(prediction[0])

# Gradio Interface
inputs = [
    gr.Slider(minimum=0, maximum=3, label="Season ('spring': 0, 'winter': 1, 'summer': 2, 'fall': 3)", step=1),
    gr.Slider(minimum=0, maximum=23, label="Hour of Day ('4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8, '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16, '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23)", step=1),
    gr.Slider(minimum=0, maximum=1, label="Holiday ('Yes': 0, 'No': 1)", step=1),
    gr.Slider(minimum=0, maximum=1, label="Working Day ('Yes': 1, 'No': 0)", step=1),
    gr.Slider(minimum=1, maximum=4, label="Weather Condition ('Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3)", step=1),
    gr.Slider(minimum=-10, maximum=40, label="Temperature (°C)"),
    gr.Slider(minimum=-10, maximum=40, label="Feels Like Temperature (°C)"),
    gr.Slider(minimum=0, maximum=100, label="Humidity (%)"),
    gr.Slider(minimum=0, maximum=50, label="Wind Speed (km/h)"),
    gr.Slider(minimum=0, maximum=1, label="Year (2011: 0, 2012: 1)", step=1),
    gr.Slider(minimum=0, maximum=11, label="Month (0=January, 11=December)", step=1),
    gr.Dropdown(choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], label="Weekday")
]

outputs = gr.Textbox(label="Predicted Bike Rentals")

# Launch Gradio Interface
app = gr.Interface(fn=predict_bike_rentals, inputs=inputs, outputs=outputs, title="Bike Rental Count Predictor")
app.launch(server_name="0.0.0.0", server_port=7860)
