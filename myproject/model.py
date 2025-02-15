import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Or the original classifier used
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier



# Load dataset
df = pd.read_csv("StressLevelDataset.csv")

# Preprocess dataset
X = df.drop(columns=["stress_level"])  # Features
y = df["stress_level"]  # Target variable

# Encode categorical labels if needed
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the updated model
with open("stress_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the Label Encoder (if categorical encoding is used)
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

print("Model retrained and saved successfully!")
import os

file_path = "stress_model.pkl"

if os.path.exists(file_path):
    print(f"File '{file_path}' exists with size {os.path.getsize(file_path)} bytes.")
else:
    print(f"File '{file_path}' does not exist.")
    # Dummy model (Replace this with your trained model)
model = DecisionTreeClassifier()
model.fit([[0, 1], [1, 0]], [0, 1])  # Example training (replace with real data)

# Save model
with open("stress_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

