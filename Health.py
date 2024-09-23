import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Path to your CSV file
file_path = r"C:\Users\Swathi\Downloads\Synthetic_Health_Monitoring_Data.csv"

# Load the dataset
data = pd.read_csv(file_path)

# Preprocessing
# Mapping activity levels to numerical values (low: 0, medium: 1, high: 2)
activity_mapping = {'low': 0, 'medium': 1, 'high': 2}
data['activity_level'] = data['activity_level'].map(activity_mapping)

# Features and target
X = data[['heart_rate', 'activity_level']]  # Features
y = data['alert']  # Target (alerts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Output accuracy and confusion matrix
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

# Plotting feature importance
features = X.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
