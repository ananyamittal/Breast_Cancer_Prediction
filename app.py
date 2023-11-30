import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.metrics import accuracy_score, precision_score, f1_score

def performance(model, x_train, y_train, y_pred, y_test):
    training_score = model.score(x_train, y_train)
    y_pred = y_pred[:len(y_test)]
    testing_score = accuracy_score(y_test, y_pred)
    return training_score, testing_score

def preprocess_data(user_input, imputer, scaler):
    user_input_imputed = pd.DataFrame(imputer.transform(user_input), columns=user_input.columns)
    user_input_scaled = pd.DataFrame(scaler.transform(user_input_imputed), columns=user_input.columns)
    return user_input_scaled

# Load the dataset
csv_file_path = 'Cancer_Data.csv'
df = pd.read_csv(csv_file_path)

# Label encoding for the 'diagnosis' column
LE = LabelEncoder()
df['diagnosis'] = LE.fit_transform(df['diagnosis'])
df['diagnosis'].unique()

# Drop 'Unnamed: 32' column
df = df.drop('Unnamed: 32', axis=1)

# Split the data
x = df.drop('diagnosis', axis=1)
y = df[['diagnosis']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)

# Standardize the data
sc_X = StandardScaler()
x_train_scaled = pd.DataFrame(sc_X.fit_transform(x_train_imputed), columns=x_train.columns)

# Logistic Regression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train_scaled, y_train.values.ravel())

# Random Forest
classifier_rf = RandomForestClassifier(random_state=0)
classifier_rf.fit(x_train_scaled, y_train.values.ravel())

# Decision Tree
classifier_dt = DecisionTreeClassifier(random_state=0)
classifier_dt.fit(x_train_scaled, y_train.values.ravel())

# Support Vector Machine (SVM)
classifier_svm = SVC(random_state=0)
classifier_svm.fit(x_train_scaled, y_train.values.ravel())

# Streamlit app
st.title("Breast Cancer Diagnosis Prediction App")

# Introduction and Instructions
st.write(
    "This app predicts whether a breast cancer tumor is malignant or benign based on various features. "
    "You can use sliders to input feature values or manually input them. The app will provide predictions using different classifiers."
    "      "
    "      "
    "      "
    )

st.title("Classifier Performance")

# Display results for Logistic Regression
st.subheader("Logistic Regression")
train_score_lr, test_score_lr = performance(classifier_lr, x_train_scaled, y_train, classifier_lr.predict(x_train_scaled), y_train)
st.write(f"Training Score: {train_score_lr}")
test_pred_lr = classifier_lr.predict(x_test)
st.write(f"Testing Score: {accuracy_score(y_test, test_pred_lr)}")
st.write(f"MAE: {mean_absolute_error(y_test, test_pred_lr)}")
st.write(f"MSE: {mean_squared_error(y_test, test_pred_lr)}")
precision_lr = precision_score(y_test, test_pred_lr)
f1_lr = f1_score(y_test, test_pred_lr)
st.write(f"Precision: {precision_lr}")
st.write(f"F1 Score: {f1_lr}")
st.write( "      ")
st.write( "      ")

# Display results for Random Forest
st.subheader("Random Forest")
train_score_rf, test_score_rf = performance(classifier_rf, x_train_scaled, y_train, classifier_rf.predict(x_train_scaled), y_test)
st.write(f"Training Score: {train_score_rf}")
st.write(f"Testing Score: {test_score_rf}")
st.write(f"MAE: {mean_absolute_error(y_test, classifier_rf.predict(x_test))}")
st.write(f"MSE: {mean_squared_error(y_test, classifier_rf.predict(x_test))}")
precision_rf = precision_score(y_test, classifier_rf.predict(x_test))
f1_rf = f1_score(y_test, classifier_rf.predict(x_test))
st.write(f"Precision: {precision_rf}")
st.write(f"F1 Score: {f1_rf}")
st.write( "      ")
st.write( "      ")

# Display results for Decision Tree
st.subheader("Decision Tree")
train_score_dt, test_score_dt = performance(classifier_dt, x_train_scaled, y_train, classifier_dt.predict(x_train_scaled), y_test)
st.write(f"Training Score: {train_score_dt}")
st.write(f"Testing Score: {test_score_dt}")
st.write(f"MAE: {mean_absolute_error(y_test, classifier_dt.predict(x_test))}")
st.write(f"MSE: {mean_squared_error(y_test, classifier_dt.predict(x_test))}")
precision_dt = precision_score(y_test, classifier_dt.predict(x_test))
f1_dt = f1_score(y_test, classifier_dt.predict(x_test))
st.write(f"Precision: {precision_dt}")
st.write(f"F1 Score: {f1_dt}")
st.write( "      ")
st.write( "      ")

# Display results for Support Vector Machine (SVM)
st.subheader("Support Vector Machine (SVM)")
train_score_svm, test_score_svm = performance(classifier_svm, x_train_scaled, y_train, classifier_svm.predict(x_train_scaled), y_test)
st.write(f"Training Score: {train_score_svm}")
st.write(f"Testing Score: {test_score_svm}")
st.write(f"MAE: {mean_absolute_error(y_test, classifier_svm.predict(x_test))}")
st.write(f"MSE: {mean_squared_error(y_test, classifier_svm.predict(x_test))}")
precision_svm = precision_score(y_test, classifier_svm.predict(x_test))
f1_svm = f1_score(y_test, classifier_svm.predict(x_test))
st.write(f"Precision: {precision_svm}")
st.write(f"F1 Score: {f1_svm}")
st.write( "      ")
st.write( "      ")

# User Input Section
st.sidebar.header("User Input for Prediction")

# Create an empty dictionary to store slider values
slider_values = {}

# Assuming x_train.columns contains the feature names
for feature in x_train.columns:
    # Add sliders for user input
    slider_min = float(x_train[feature].min())
    slider_max = float(x_train[feature].max())
    slider_mean = float(x_train[feature].mean())
    
    # Use the feature name as the key for the dictionary
    slider_values[feature] = st.sidebar.slider(f"{feature} Slider", slider_min, slider_max, slider_mean)

# Create a DataFrame from the dictionary
user_input = pd.DataFrame([slider_values])

# Add a text box for user input
user_input_text = st.sidebar.text_area("Enter comma-separated values for features (e.g., 13.54, 14.36, 87.46, 566.3, 0.09779, ...):")
if user_input_text:
    # Create a DataFrame from the user input text
    user_input_manual = pd.DataFrame([user_input_text.split(',')], columns=x_train.columns)
    user_input_manual = user_input_manual.astype(float)

    # Update the slider_values dictionary with manual input
    for feature in x_train.columns:
        if feature in user_input_manual.columns:
            slider_values[feature] = user_input_manual[feature].values[0]

# Create a DataFrame from the dictionary
user_input = pd.DataFrame([slider_values])

# Preprocess user input
user_input_scaled = preprocess_data(user_input, imputer, sc_X)

# Logistic Regression Prediction
lr_prediction = classifier_lr.predict(user_input_scaled)
st.sidebar.subheader("Logistic Regression Prediction:")
st.sidebar.write(f"Prediction: {lr_prediction[0]} - {'Malignant' if lr_prediction[0] == 1 else 'Benign'}")

# Random Forest Prediction
rf_prediction = classifier_rf.predict(user_input_scaled)
st.sidebar.subheader("Random Forest Prediction:")
st.sidebar.write(f"Prediction: {rf_prediction[0]} - {'Malignant' if rf_prediction[0] == 1 else 'Benign'}")

# Decision Tree Prediction
dt_prediction = classifier_dt.predict(user_input_scaled)
st.sidebar.subheader("Decision Tree Prediction:")
st.sidebar.write(f"Prediction: {dt_prediction[0]} - {'Malignant' if dt_prediction[0] == 1 else 'Benign'}")

# SVM Prediction
svm_prediction = classifier_svm.predict(user_input_scaled)
st.sidebar.subheader("SVM Prediction:")
st.sidebar.write(f"Prediction: {svm_prediction[0]} - {'Malignant' if svm_prediction[0] == 1 else 'Benign'}")
