"""
Multiple Linear Regression Model for Campus Energy Waste Prediction
Includes training, evaluation, and visualization functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle


def prepare_features(df):
    """Encode categorical variables and prepare feature matrix."""
    df_encoded = df.copy()
    
    # Encode categoricals
    le_room = LabelEncoder()
    le_time = LabelEncoder()
    le_day = LabelEncoder()
    
    df_encoded['room_type_enc'] = le_room.fit_transform(df['room_type'])
    df_encoded['time_slot_enc'] = le_time.fit_transform(df['time_slot'])
    df_encoded['day_enc'] = le_day.fit_transform(df['day_of_week'])
    
    # Select features
    feature_cols = [
        'capacity', 'scheduled_students', 'actual_attendance',
        'year_of_study', 'duration_hours', 'weather_index',
        'utilization_ratio', 'empty_seats',
        'room_type_enc', 'time_slot_enc', 'day_enc'
    ]
    
    X = df_encoded[feature_cols]
    y = df_encoded['energy_waste']
    
    return X, y, feature_cols


def train_model(X, y):
    """Train Multiple Linear Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    return metrics, y_pred


def plot_predicted_vs_actual(y_test, y_pred, save_path=None):
    """Create Predicted vs Actual scatter plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#4a7c43', edgecolors='white', s=40)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Energy Waste', fontsize=12)
    plt.ylabel('Predicted Energy Waste', fontsize=12)
    plt.title('Predicted vs Actual Energy Waste', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt.gcf()


def plot_coefficients(model, feature_names, save_path=None):
    """Create coefficient importance bar chart."""
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = ['#f44336' if c < 0 else '#4a7c43' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title('Feature Coefficients (Impact on Energy Waste)', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt.gcf()


def get_insights(model, feature_names):
    """Generate sustainability insights from coefficients."""
    coef_dict = dict(zip(feature_names, model.coef_))
    
    insights = []
    
    if coef_dict.get('empty_seats', 0) > 0:
        insights.append(f"Each empty seat adds ~{coef_dict['empty_seats']:.3f} energy waste units")
    
    if coef_dict.get('duration_hours', 0) > 0:
        insights.append(f"Each hour of class adds ~{coef_dict['duration_hours']:.2f} waste units")
    
    if coef_dict.get('utilization_ratio', 0) < 0:
        insights.append("Higher room utilization significantly reduces energy waste")
    
    insights.append("Recommendation: Match room capacity to expected attendance")
    insights.append("Recommendation: Prioritize morning scheduling to reduce cooling load")
    
    return insights


if __name__ == "__main__":
    from data_generation import generate_dataset
    
    print("Loading dataset...")
    df = generate_dataset()
    
    print("Preparing features...")
    X, y, feature_cols = prepare_features(df)
    
    print("Training model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    print("Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    
    print("\nSustainability Insights:")
    for insight in get_insights(model, feature_cols):
        print(f"  • {insight}")
    
    # Save plots
    plot_predicted_vs_actual(y_test, y_pred, 'predicted_vs_actual.png')
    plot_coefficients(model, feature_cols, 'coefficients.png')
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)
    
    print("\nModel and plots saved.")
