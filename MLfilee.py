import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import random
import time

# -------------------------------
# Module 1: Simulated Sensor Data
# -------------------------------

class WasteSensorSimulator:
    def __init__(self, bin_id, location):
        self.bin_id = bin_id
        self.location = location
        self.fill_level = random.randint(10, 40)
        self.history = []

    def update_fill_level(self):
        increment = random.randint(5, 15)
        self.fill_level = min(100, self.fill_level + increment)
        timestamp = datetime.now()
        self.history.append((timestamp, self.fill_level))
        return timestamp, self.fill_level

    def get_history_df(self):
        return pd.DataFrame(self.history, columns=['Timestamp', 'Fill_Level'])

# -------------------------------
# Module 2: Predictive Analytics
# -------------------------------

class FillLevelPredictor:
    def __init__(self, history_df):
        self.df = history_df.copy()
        self.model = LinearRegression()

    def train_model(self):
        self.df['Day'] = (self.df['Timestamp'] - self.df['Timestamp'].min()).dt.days
        X = self.df[['Day']]
        y = self.df['Fill_Level']
        self.model.fit(X, y)

    def predict_next_days(self, days=5):
        last_day = self.df['Day'].max()
        future_days = pd.DataFrame({'Day': list(range(last_day + 1, last_day + days + 1))})
        predictions = self.model.predict(future_days)
        future_days['Predicted_Fill_Level'] = predictions
        return future_days

# -------------------------------
# Module 3: Smart Scheduling
# -------------------------------

class CollectionScheduler:
    def __init__(self, threshold=85):
        self.threshold = threshold

    def generate_schedule(self, predictions_df):
        alerts = predictions_df[predictions_df['Predicted_Fill_Level'] > self.threshold]
        schedule = []
        for _, row in alerts.iterrows():
            day_offset = int(row['Day'])
            collection_date = datetime.now() + timedelta(days=day_offset)
            schedule.append((collection_date.date(), row['Predicted_Fill_Level']))
        return schedule

# -------------------------------
# Module 4: Recycling Awareness
# -------------------------------

class RecyclingAssistant:
    def __init__(self):
        self.tips = {
            'plastic': "♻️ Rinse plastic containers before recycling. Avoid single-use plastics.",
            'paper': "📄 Flatten cardboard boxes. Keep paper dry and clean.",
            'glass': "🍾 Remove lids and rinse glass bottles. Don't mix with ceramics.",
            'organic': "🌱 Compost food scraps and garden waste. Avoid mixing with plastics.",
            'metal': "🔩 Clean metal cans. Avoid mixing with hazardous waste."
        }

    def get_tip(self, material):
        return self.tips.get(material.lower(), "No tips available for this material.")

# -------------------------------
# Module 5: Dashboard Simulation
# -------------------------------

class WasteDashboard:
    def __init__(self, sensor, predictor, scheduler, assistant):
        self.sensor = sensor
        self.predictor = predictor
        self.scheduler = scheduler
        self.assistant = assistant

    def display_status(self):
        print(f"\n📍 Bin ID: {self.sensor.bin_id} | Location: {self.sensor.location}")
        print(f"Current Fill Level: {self.sensor.fill_level}%")

    def display_predictions(self):
        self.predictor.train_model()
        future_df = self.predictor.predict_next_days()
        print("\n📈 Predicted Fill Levels:")
        print(future_df)
        return future_df

    def display_schedule(self, future_df):
        schedule = self.scheduler.generate_schedule(future_df)
        if schedule:
            print("\n🗓️ Scheduled Collections:")
            for date, level in schedule:
                print(f"{date}: Predicted fill level = {level:.2f}%")
        else:
            print("\n✅ No collection needed in the next few days.")

    def display_tips(self):
        print("\n💡 Recycling Tips:")
        for material in self.assistant.tips.keys():
            print(f"{material.capitalize()}: {self.assistant.get_tip(material)}")

    def plot_fill_trend(self):
        df = self.sensor.get_history_df()
        self.predictor.train_model()
        future_df = self.predictor.predict_next_days()

        plt.figure(figsize=(10, 5))
        plt.plot(df['Timestamp'], df['Fill_Level'], label='Actual Fill Level')
        future_dates = [df['Timestamp'].max() + timedelta(days=int(d)) for d in future_df['Day']]
        plt.plot(future_dates, future_df['Predicted_Fill_Level'], label='Predicted Fill Level', linestyle='--')
        plt.axhline(y=self.scheduler.threshold, color='r', linestyle=':', label='Overflow Threshold')
        plt.xlabel('Date')
        plt.ylabel('Fill Level (%)')
        plt.title('Smart Waste Bin Fill Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# -------------------------------
# Main Execution
# -------------------------------

def main():
    # Initialize modules
    sensor = WasteSensorSimulator(bin_id="BIN-001", location="Sector 12, GreenCity")
    assistant = RecyclingAssistant()
    scheduler = CollectionScheduler()

    # Simulate data collection
    print("🔄 Simulating sensor data...")
    for _ in range(10):
        timestamp, level = sensor.update_fill_level()
        print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - Fill Level: {level}%")
        time.sleep(0.1)  # Simulate delay

    # Predict and display dashboard
    history_df = sensor.get_history_df()
    predictor = FillLevelPredictor(history_df)
    dashboard = WasteDashboard(sensor, predictor, scheduler, assistant)

    dashboard.display_status()
    future_df = dashboard.display_predictions()
    dashboard.display_schedule(future_df)
    dashboard.display_tips()
    dashboard.plot_fill_trend()

if __name__ == "__main__":
    main()