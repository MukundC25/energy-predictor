# Campus Energy Waste Predictor

A Multiple Linear Regression system that predicts hidden energy waste caused by academic scheduling inefficiencies.

## Problem Statement

Academic institutions waste significant energy due to:
- Empty seats in classrooms (unused lighting, cooling)
- Room-capacity mismatches
- Predictable low-attendance periods (afternoons, Fridays, senior years)

This project quantifies these inefficiencies using interpretable regression.

## Project Structure

```
campus-energy-predictor/
├── data_generation.py    # Synthetic dataset generator (~7000 rows)
├── regression_model.py   # MLR training and evaluation
├── streamlit_app.py      # Web interface for predictions
└── README.md
```

## Dataset Features

| Feature | Description |
|---------|-------------|
| capacity | Room seating capacity |
| scheduled_students | Students registered |
| actual_attendance | Students who attended |
| year_of_study | 1-4 (senior attendance is lower) |
| time_slot | morning/afternoon/evening |
| duration_hours | Class length |
| weather_index | Cooling load proxy |
| **energy_waste** | Target: waste units |

## Key Findings

- **Empty seats** are the primary waste driver (~0.08 units per seat/hour)
- **Afternoon classes** incur cooling penalties
- **Senior students** have 40% lower attendance than freshmen
- **Room-capacity matching** can reduce waste by 15-25%

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib plotly streamlit
```

## Usage

**Generate data and train model:**
```bash
python regression_model.py
```

**Run Streamlit app:**
```bash
streamlit run streamlit_app.py
```

## Model Performance

- R² Score: ~0.93
- RMSE: ~1.0 units
- MAE: ~0.8 units

## Sustainability Recommendations

1. Match room capacity to expected attendance
2. Prioritize morning scheduling
3. Consolidate classes during low-attendance periods
4. Reserve labs for hands-on sessions only

---

*Applied Artificial Intelligence for Sustainability - Academic Project*
