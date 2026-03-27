import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

class AirfoilAI:
    def __init__(self, csv_path=None):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        if csv_path:
            self.train(csv_path)

    def train(self, csv_path):
        df = pd.read_csv(csv_path)
        
        X = df[['velocity', 'altitude']]
        y = df[['best_m', 'best_p', 'best_t']]
        
        self.model.fit(X, y)
        print("AI finished training on dataset.")

    def predict_naca(self, velocity, altitude):
        input_data = pd.DataFrame([[velocity, altitude]], columns=['velocity', 'altitude'])

        pred = self.model.predict(input_data)[0]

        m = int(round(pred[0]))
        p = int(round(pred[1]))
        t = int(round(pred[2]))

        return f"{m}{p}{t:02d}"

    def save_model(self, filename="../models/airfoil_model.pkl"):
        joblib.dump(self.model, filename)
        