import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


dev_data_path = './Dev_data_to_be_shared.csv'
dev_data = pd.read_csv(dev_data_path)


dev_data['bureau_436'] = dev_data['bureau_447'].fillna(0)  # Fill specific column as done in the original code
X = dev_data.drop(columns=['bad_flag', 'account_number'], axis=1)
y = dev_data['bad_flag']


imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f"ROC-AUC Score: {roc_auc}")
print(classification_report(y_test, y_pred))


feature_names = X.columns
class_names = ['Not Bad', 'Bad']

explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    training_labels=y_train
)


instance_index = 0  # Change this to explore other instances
exp = explainer.explain_instance(
    data_row=X_test[instance_index],
    predict_fn=model.predict_proba
)

exp.show_in_notebook(show_table=True)

exp.save_to_file('lime_explanation.html')

validation_data_path = './Validation_data_to_be_shared.csv'
validation_data = pd.read_csv(validation_data_path)

validation_data['bureau_436'] = validation_data['bureau_447'].fillna(0)  # Consistent with dev_data processing
validation_features = validation_data.drop(columns=['account_number'], axis=1)

validation_features_imputed = imputer.transform(validation_features)

validation_features_scaled = scaler.transform(validation_features_imputed)

validation_data['predicted_probability'] = model.predict_proba(validation_features_scaled)[:, 1]

submission = validation_data[['account_number', 'predicted_probability']]
submission.to_csv('Predicted_Probabilities.csv', index=False)

print("Predictions saved to 'Predicted_Probabilities.csv'.")





predictions_path = './Predicted_Probabilities.csv'  # Path to your predictions CSV file
predictions = pd.read_csv(predictions_path)


print(predictions.head())



# Visualization of predicted probabilities
plt.hist(predictions['predicted_probability'], bins=50, alpha=0.75, color='blue')
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Predicted Probability of Default")
plt.ylabel("Frequency")
plt.show()


threshold = 0.5
predictions['predicted_class'] = (predictions['predicted_probability'] > threshold).astype(int)


print("Summary of Predicted Probabilities:")
print(predictions['predicted_probability'].describe())

# Optionally, save the updated predictions with the classes back to a CSV file
predictions.to_csv('Validated_Predictions.csv', index=False)

print("Validation completed. Predictions saved to 'Validated_Predictions.csv'.")