import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE

# Load the data from the .mat file
data = scipy.io.loadmat('new_structure_data.mat')

# Extract the records with key name "VibData_x"
vibration_data = []
people_id = []
environment_id = []
sensor_id = []
lane_id = []
shoe_id = []

for key in data:
    if key.startswith('VibData_'):
        record = data[key]
        vibration_data.append(record['vibration_data'][0][0])
        people_id.append(record['People_id'][0][0])
        environment_id.append(record['Environment_id'][0][0])
        sensor_id.append(record['Sensor_id'][0][0])
        lane_id.append(record['Lane_id'][0][0])
        shoe_id.append(record['Shoe_id'][0][0])

# Create a DataFrame for easier manipulation
df = pd.DataFrame({
    'Vibration_data': vibration_data,
    'people_id': people_id,
    'environment_id': environment_id,
    'sensor_id': sensor_id,
    'lane_id': lane_id,
    'shoe_id': shoe_id
})

# Pre-select a subset of relevant features
selected_features = pd.DataFrame()

# Extract features from Vibration_data (e.g., mean, std, max, min)
selected_features['mean'] = df['Vibration_data'].apply(lambda x: np.mean(x))
selected_features['std'] = df['Vibration_data'].apply(lambda x: np.std(x))
selected_features['max'] = df['Vibration_data'].apply(lambda x: np.max(x))
selected_features['min'] = df['Vibration_data'].apply(lambda x: np.min(x))

# Print the selected features to check the extraction
print("Selected features head:\n", selected_features.head(10))

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features)

# Use people_id directly as labels
labels = np.array([label[0] for label in df['people_id']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, labels, test_size=0.3, random_state=42
)

# Apply RFE (Recursive Feature Elimination) with SVM using a linear kernel
svm_model = SVC(kernel='linear', random_state=42)
selector = RFE(svm_model, n_features_to_select=2)  # Select 2 best features
selector = selector.fit(X_train, y_train)

# Get the selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train the SVM model on the selected features
svm_model.fit(X_train_selected, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_selected)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the decision boundary (only for 2D data, using selected features for visualization)
plt.figure(figsize=(10, 6))

# Generate a meshgrid for plotting (use only the selected 2 features)
x_min, x_max = X_train_selected[:, 0].min() - 1, X_train_selected[:, 0].max() + 1
y_min, y_max = X_train_selected[:, 1].min() - 1, X_train_selected[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# Make predictions for the meshgrid to plot the decision boundary
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(
    X_train_selected[:, 0], X_train_selected[:, 1], c=y_train, edgecolors='k', cmap='coolwarm', label="Train"
)
plt.scatter(
    X_test_selected[:, 0], X_test_selected[:, 1], c=y_test, edgecolors='k', cmap='coolwarm', marker='x', label="Test"
)

plt.title("SVM Decision Boundary with RFE Feature Selection")
plt.xlabel("Selected Feature 1")
plt.ylabel("Selected Feature 2")
plt.legend()
plt.show()
