# AI-Driven Airfoil Optimization & Prediction Engine ✈️

This project implements a high-performance pipeline for aerodynamic shape optimization. It uses **Bayesian Optimization** to discover ideal NACA 4-digit airfoils across a flight envelope and trains a **Random Forest Surrogate Model** to provide instantaneous aerodynamic predictions.

## 🚀 Key Features

* **Automated Optimization:** Powered by `Optuna` to minimize drag and maximize lift-to-drag ratios ($L/D$).
* **Physics-Backed:** Integrated with `XFoil` / `AeroSandbox` for high-fidelity aerodynamic polar generation.
* **Neural-Speed Inference:** Replaces slow iterative solvers with a trained AI model capable of sub-millisecond predictions.
* **Visual Analysis:** Built-in tools for generating Performance Heatmaps (Finesse, Camber, and Thickness).

---

## 🏗️ System Architecture

The project follows a modular "Generator-Optimizer-Predictor" workflow:

```mermaid
graph LR
    A[Geometry Engine] --> B[Aero Solver]
    B --> C[Optuna Optimizer]
    C --> D[(Training Dataset)]
    D --> E[StandardScaler]
    E --> F[Random Forest Model]
    F --> G[Real-time Prediction]
```  

---

## 📦 Installation

Clone the repository:

```Bash
git clone [https://github.com/your-username/airfoil-ai-opt.git](https://github.com/your-username/airfoil-ai-opt.git)
cd airfoil-ai-opt
Set up a virtual environment:
```

```Bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install dependencies:
```

```Bash
pip install -r requirements.txt
🛠️ Usage
```

1. Run Optimization
Generate the dataset by running the Optuna study. This will explore the design space and save results to data/.

```Python
python main_optimization.py
```

2. Train the AI
Process the generated data and "freeze" the model.

```Python
from src.airfoil_predictor import AirfoilAI
ai = AirfoilAI("data/optimization_results.csv")
ai.save_model()
```

3. Predict & Visualize
Get the optimal NACA profile for any flight condition:

```Python
naca = ai.predict_naca(velocity=35, altitude=1500)
print(f"Optimal Profile: {naca}")
```

---

## 📊 Performance Visualization

The system includes a plotting suite to validate the AI's learning. Below is an example of the Finesse Map, showing how aerodynamic efficiency evolves across the flight envelope.

---

## 📂 Project Structure

src/ : Core logic (Geometry, Solver, Predictor).

data/ : CSV datasets and serialized .pkl models.

notebooks/ : Exploratory data analysis and plotting.

requirements.txt : Python dependencies.

.gitignore : Ensures a clean repository.

---

## ⚖️ License

Distributed under the MIT License. See LICENSE for more information.
