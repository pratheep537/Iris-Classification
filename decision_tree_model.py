# decision_tree_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("IRIS.csv")

# Encode labels
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# Split features and labels
X = df.drop("species", axis=1)
y = df["species"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model and label encoder
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
