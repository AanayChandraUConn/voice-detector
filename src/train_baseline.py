import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset we built on Day 3
df = pd.read_csv("data/features.csv")

print(f"Loaded {len(df)} rows")
print(df["label"].value_counts())

# Separate the 13 numbers (X) from the answer key (y)
X = df.drop("label", axis=1)
y = df["label"]

# Split into training data (model learns from this) and test data (model is judged on this)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} clips, testing on {len(X_test)} clips")


# Train a simple baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test it on the held-out data it's never seen
predictions = model.predict(X_test)

# Check how well it did
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.2%}")
print("\nDetailed results:")
print(classification_report(y_test, predictions))
