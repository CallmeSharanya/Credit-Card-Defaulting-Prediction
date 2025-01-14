import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# Step 1: Load the Development Data
dev_data_path = './Dev_data_to_be_shared.csv'
dev_data = pd.read_csv(dev_data_path)

# Handle missing values in the development data
dev_data['bureau_436'] = dev_data['bureau_447'].fillna(0)  # Fill specific column as done in the original code
X = dev_data.drop(columns=['bad_flag', 'account_number'], axis=1)
y = dev_data['bad_flag']

# Impute missing values using median
imputer = SimpleImputer(strategy='median')  # You can change the strategy if needed
X_imputed = imputer.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

# Step 2: Split the Development Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f"ROC-AUC Score: {roc_auc}")
print(classification_report(y_test, y_pred))

# Step 5: Load Validation Data and Predict Probabilities
validation_data_path = './Validation_data_to_be_shared.csv'
validation_data = pd.read_csv(validation_data_path)

# Preprocess validation data
validation_data['bureau_436'] = validation_data['bureau_447'].fillna(0)  # Consistent with dev_data processing
validation_features = validation_data.drop(columns=['account_number'], axis=1)

# Impute missing values in validation data using the same SimpleImputer
validation_features_imputed = imputer.transform(validation_features)

# Standardize validation data using the fitted StandardScaler
validation_features_scaled = scaler.transform(validation_features_imputed)

# Predict probabilities
validation_data['predicted_probability'] = model.predict_proba(validation_features_scaled)[:, 1]

# Step 6: Prepare Submission File
submission = validation_data[['account_number', 'predicted_probability']]
submission.to_csv('Predicted_Probabilities.csv', index=False)

print("Predictions saved to 'Predicted_Probabilities.csv'.")
