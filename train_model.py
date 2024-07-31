import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv(r'C:\Users\ASIF UMAR\Documents\GitHub\As_project\diabetes.csv')

# Define features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Check for missing values
print(df.isnull().sum())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)  # Increased iterations for convergence
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, r'C:\Users\ASIF UMAR\Documents\GitHub\As_project\diabetes_model.pkl')
joblib.dump(scaler, r'C:\Users\ASIF UMAR\Documents\GitHub\As_project\scaler.pkl')
