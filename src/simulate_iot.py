import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# 1. Generate 1000 rows of normal sensor data
np.random.seed(42)
rows = 1000
data = {
    'temperature': np.random.uniform(40, 70, rows), # Normal temp
    'vibration': np.random.uniform(1, 3, rows),     # Normal vibration
    'current': np.random.uniform(5, 12, rows)       # Normal current
}

df = pd.DataFrame(data)

# 2. Inject "Failure" scenarios (The Anomalies)
# Let's make the last 150 rows represent a machine breaking down
df.loc[850:, 'temperature'] = np.random.uniform(85, 110, 150) # Overheating
df.loc[850:, 'vibration'] = np.random.uniform(5, 8, 150)     # Heavy shaking

# 3. Create the Target Label (Logic the AI must learn)
# Failure (1) occurs if Temp > 80 AND Vibration > 4
df['failure'] = ((df['temperature'] > 80) & (df['vibration'] > 4)).astype(int)

# 4. Save to CSV
df.to_csv("data/iot_sensor_data.csv", index=False)
print("✅ Dataset generated with 'Failure' cases successfully!")
print("Total Failures in Data: {df['failure'].sum()}")