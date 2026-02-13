import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

np.random.seed(42)

n = 800

cgpa = np.random.uniform(5, 10, n)
internship = np.random.randint(0, 2, n)
projects = np.random.randint(0, 6, n)
skill_match = np.random.uniform(0, 100, n)
communication = np.random.uniform(4, 10, n)

# FINAL BALANCED FORMULA
placement_score = (
    0.4 * cgpa +
    1.2 * internship +
    0.8 * projects +
    0.03 * skill_match +
    0.4 * communication
)

placed = (placement_score > 9).astype(int)

data = pd.DataFrame({
    "CGPA": cgpa,
    "Internship": internship,
    "Projects": projects,
    "SkillMatch": skill_match,
    "Communication": communication,
    "Placed": placed
})

X = data.drop("Placed", axis=1)
y = data["Placed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

joblib.dump(model, "placement_model.pkl")
print("Balanced model saved as placement_model.pkl")