import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import optuna
from src.geometry import AirfoilGeometry
from src.physics import AeroSolver
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AirfoilAI:
    def __init__(self, model_path=None, scaler_path=None):
        self.model = None
        self.scaler = None
        
        if model_path and scaler_path:
            self.load_model(model_path, scaler_path)
        else:
            # Default state for new training
            self.model = RandomForestRegressor(n_estimators=200, random_state=42)
            self.scaler = StandardScaler()

    def load_model(self, model_path, scaler_path):
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"AI Loaded from {os.path.dirname(model_path)}")
        else:
            print("⚠️ Files not found. Ensure both .pkl files exist.")

    def train(self, csv_path):
        df = pd.read_csv(csv_path)
        X = df[['velocity', 'altitude']]
        y = df[['best_m', 'best_p', 'best_t']]
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        print("✅ AI finished training with standardized features.")

    def predict_naca(self, velocity, altitude):
        if self.model is None or self.scaler is None:
            raise Exception("AI not initialized.")

        input_data = pd.DataFrame([[velocity, altitude]], columns=['velocity', 'altitude'])
        
        input_scaled = self.scaler.transform(input_data)
        
        pred = self.model.predict(input_scaled)[0]
        m, p, t = int(round(pred[0])), int(round(pred[1])), int(round(pred[2]))
        return f"{m}{p}{t:02d}"

    def save_model(self, folder="../models"):
        """Saves both model and scaler to the specified folder."""
        os.makedirs(folder, exist_ok=True)
        
        model_file = os.path.join(folder, "airfoil_model.pkl")
        scaler_file = os.path.join(folder, "scaler.pkl")
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        print(f"Saved: {model_file} and {scaler_file}")
        
class AirfoilOptimizer:
    def __init__(self, chord=1.0, alpha=0.0):
        self.chord = chord
        self.alpha = alpha

    def _objective(self, trial, v_target, alt_target):
        """The internal objective function that Optuna will call."""
        m = trial.suggest_int("m", 0, 9)
        p = trial.suggest_int("p", 0, 9)
        t = trial.suggest_int("t", 1, 40)
        
        af = AirfoilGeometry(n_points=100)
        xu, yu, xl, yl = af.generate_naca4(m_int=m, p_int=p, t_int=t)
        x_coords, y_coords = af.get_coords_for_solver(xu, yu, xl, yl)
        
        solver = AeroSolver(altitude=alt_target, velocity=v_target, chord=self.chord)
        res = solver.solve(x_coords, y_coords, alpha=self.alpha)
        
        cl = res['cl'][0]
        cd = res['cd'][0]
        
        if cd <= 1e-6 or cl <= 0:
            return 0
        
        return cl / cd

    def find_best_airfoil(self, v_target, alt_target, n_trials=50):
        """Orchestrates the Optuna study for a specific condition."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        
        study.optimize(lambda t: self._objective(t, v_target, alt_target), n_trials=n_trials)
        
        return study.best_params, study.best_value