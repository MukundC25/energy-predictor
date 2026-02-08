"""
Synthetic Campus Energy Waste Dataset Generator
Generates ~7000 rows of academic scheduling data for energy waste prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Campus configuration
BUILDINGS = 5
ROOMS_PER_BUILDING = 12
DAYS = 100  # Academic days to simulate

# Room types with capacities
ROOM_TYPES = {
    'small_classroom': (30, 45),
    'large_classroom': (60, 90),
    'lab': (25, 40)
}

# Time slots
TIME_SLOTS = ['morning', 'afternoon', 'evening']
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

# Attendance factors (behavioral patterns)
TIME_FACTORS = {'morning': 0.90, 'afternoon': 0.75, 'evening': 0.65}
DAY_FACTORS = {'Monday': 0.88, 'Tuesday': 0.92, 'Wednesday': 0.90, 
               'Thursday': 0.85, 'Friday': 0.70, 'Saturday': 0.55}
YEAR_FACTORS = {1: 0.92, 2: 0.85, 3: 0.75, 4: 0.60}


def generate_rooms():
    """Generate room inventory for the campus."""
    rooms = []
    for building in range(1, BUILDINGS + 1):
        for room_num in range(1, ROOMS_PER_BUILDING + 1):
            room_type = np.random.choice(list(ROOM_TYPES.keys()), p=[0.4, 0.35, 0.25])
            capacity = np.random.randint(*ROOM_TYPES[room_type])
            rooms.append({
                'building_id': building,
                'room_id': f"B{building}R{room_num}",
                'room_type': room_type,
                'capacity': capacity
            })
    return pd.DataFrame(rooms)


def calculate_attendance(scheduled, time_slot, day, year):
    """Calculate actual attendance based on behavioral patterns."""
    base_rate = 0.75
    rate = base_rate * TIME_FACTORS[time_slot] * DAY_FACTORS[day] * YEAR_FACTORS[year]
    rate *= np.random.uniform(0.85, 1.15)  # Add noise
    actual = int(scheduled * min(rate, 1.0))
    return max(actual, int(scheduled * 0.2))  # Minimum 20% attendance


def calculate_energy_waste(row):
    """
    Calculate energy waste units based on room utilization.
    Key factors: empty seats, room type, time slot, duration.
    """
    empty_seats = row['capacity'] - row['actual_attendance']
    duration = row['duration_hours']
    
    # Base waste from empty seats
    waste = empty_seats * 0.08 * duration
    
    # Lab premium (equipment running)
    if row['room_type'] == 'lab':
        waste *= 1.4
    
    # Afternoon cooling load
    if row['time_slot'] == 'afternoon':
        waste += 0.5 * duration
    elif row['time_slot'] == 'evening':
        waste += 0.3 * duration
    
    # Weather impact (simulated)
    waste *= (1 + row['weather_index'] * 0.3)
    
    return round(waste, 2)


def generate_dataset():
    """Generate the complete campus energy waste dataset."""
    rooms = generate_rooms()
    records = []
    
    start_date = datetime(2025, 8, 1)
    
    for day_offset in range(DAYS):
        current_date = start_date + timedelta(days=day_offset)
        if current_date.weekday() == 6:  # Skip Sundays
            continue
            
        day_name = DAYS_OF_WEEK[current_date.weekday()]
        
        for _, room in rooms.iterrows():
            # Each room has 1-2 classes per day
            num_classes = np.random.choice([1, 2], p=[0.6, 0.4])
            
            for _ in range(num_classes):
                time_slot = np.random.choice(TIME_SLOTS, p=[0.45, 0.35, 0.20])
                year = np.random.choice([1, 2, 3, 4], p=[0.30, 0.28, 0.24, 0.18])
                scheduled = min(np.random.randint(15, room['capacity'] + 10), room['capacity'])
                actual = calculate_attendance(scheduled, time_slot, day_name, year)
                duration = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                weather = np.random.uniform(0.2, 0.9)
                
                record = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'building_id': room['building_id'],
                    'room_type': room['room_type'],
                    'capacity': room['capacity'],
                    'scheduled_students': scheduled,
                    'actual_attendance': actual,
                    'year_of_study': year,
                    'time_slot': time_slot,
                    'day_of_week': day_name,
                    'duration_hours': duration,
                    'weather_index': round(weather, 2)
                }
                records.append(record)
    
    df = pd.DataFrame(records)
    
    # Add derived features
    df['utilization_ratio'] = df['actual_attendance'] / df['capacity']
    df['empty_seats'] = df['capacity'] - df['actual_attendance']
    
    # Calculate target variable
    df['energy_waste'] = df.apply(calculate_energy_waste, axis=1)
    
    return df


if __name__ == "__main__":
    print("Generating Campus Energy Waste Dataset...")
    df = generate_dataset()
    df.to_csv('campus_data.csv', index=False)
    print(f"Generated {len(df)} records")
    print(f"Energy waste range: {df['energy_waste'].min():.2f} - {df['energy_waste'].max():.2f}")
    print(f"Mean energy waste: {df['energy_waste'].mean():.2f}")
