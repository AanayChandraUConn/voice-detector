import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/features.csv")

print(f"Loaded {len(df)} rows")
print(df["label"].value_counts())

X = df.drop("label", axis=1)
y = df["label"]

# 80/20 split, model never sees the test 20% during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} clips, testing on {len(X_test)} clips")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.2%}")
print("\nDetailed results:")
print(classification_report(y_test, predictions))
