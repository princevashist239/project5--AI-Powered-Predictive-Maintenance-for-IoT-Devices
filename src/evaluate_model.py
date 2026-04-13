import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Load data and model [cite: 522, 550]
data = pd.read_csv("data/iot_sensor_data.csv")
model = joblib.load("models/predictive_maintenance_model.pkl")

# 2. Generate predictions [cite: 532]
X = data[['temperature', 'vibration', 'current']]
y_true = data['failure']
y_pred = model.predict(X)

# 3. Create a Confusion Matrix [cite: 234]
# Adding labels=[0, 1] ensures the 2x2 grid always appears
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Failure'], 
            yticklabels=['Normal', 'Failure'])

plt.title('AI Predictive Maintenance: Performance Analysis')
plt.ylabel('Actual Machine State')
plt.xlabel('AI Predicted State')

# 4. Save the output for GitHub 
plt.savefig('outputs/model_performance.png')
print("✅ Visualization saved to outputs/model_performance.png")