import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the red wine dataset with the correct delimiter
redWine_df = pd.read_csv(r'C:\Users\sinve\Semester 5\MachineLearning\WineQuality\data\winequality-red.csv', delimiter=';')

# Define multiple quality categories based on quality scores
def quality_label(quality):
    if quality <= 3:
        return "Very Low Quality"
    elif quality <= 5:
        return "Low Quality"
    elif quality == 6:
        return "Average Quality"
    elif quality == 7:
        return "Good Quality"
    else:
        return "Excellent Quality"

# Apply the function to create a new target column
redWine_df['quality_class'] = redWine_df['quality'].apply(quality_label)

# Initialize LabelEncoder
label_encoder = LabelEncoder()
redWine_df['quality_class_encoded'] = label_encoder.fit_transform(redWine_df['quality_class'])

# Separate features and target variable
X_red = redWine_df.drop(['quality', 'quality_class', 'quality_class_encoded'], axis=1)
y_red = redWine_df['quality_class_encoded']

# Train-test split
X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(X_red, y_red, test_size=0.2, random_state=42)


# Train the Random Forest model
final_rf_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='log2',
    max_depth=15,
    bootstrap=True,
    random_state=42
)
final_rf_model.fit(X_red_train, y_red_train)
# I ended up with this model due to heighest accurate metrics early on
# By continously testing new parameters and using grid search, i found two sets of parameters.
# I chose this one because i prioritized generalization over more accurate metrics(accuracy,recall,f1,score etc)

# Save the model and the label encoder
with open('WineQuality/model.pkl', 'wb') as file:
    pickle.dump(final_rf_model, file)

with open('WineQuality/label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
