import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('Titanic.csv')

# Preprocess the data (simplified version)
data = data.dropna(subset=['Age', 'Sex', 'Fare'])
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Split the data
X = data[['Age', 'Sex', 'Fare']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Save the trained model
joblib.dump(model, 'model.pkl')
