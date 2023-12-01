import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score

def performance(model, x_train, y_train, x_test, y_test):
    training_score = model.score(x_train, y_train)
    testing_score = model.score(x_test, y_test)
    return training_score, testing_score

def preprocess_data(user_input, scaler):
    user_input_scaled = pd.DataFrame(scaler.transform(user_input), columns=user_input.columns)
    return user_input_scaled

csv_file_path = 'Cancer_Data.csv'
df = pd.read_csv(csv_file_path)

LE = LabelEncoder()
df['diagnosis'] = LE.fit_transform(df['diagnosis'])
df['diagnosis'].unique()

df = df.drop('Unnamed: 32', axis=1)

x = df.drop('diagnosis', axis=1)
y = df['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Impute missing values directly using fillna
x_train_imputed = x_train.fillna(x_train.mean())
x_test_imputed = x_test.fillna(x_test.mean())

sc_X = StandardScaler()
x_train_scaled = pd.DataFrame(sc_X.fit_transform(x_train_imputed), columns=x_train.columns)
x_test_scaled = pd.DataFrame(sc_X.transform(x_test_imputed), columns=x_test.columns)

classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train_scaled, y_train)

classifier_rf = RandomForestClassifier(random_state=0)
classifier_rf.fit(x_train_scaled, y_train)

classifier_dt = DecisionTreeClassifier(random_state=0)
classifier_dt.fit(x_train_scaled, y_train)

classifier_svm = SVC(random_state=0)
classifier_svm.fit(x_train_scaled, y_train)

st.title("Breast Cancer Diagnosis Prediction App")

st.write(
    "This app predicts whether a breast cancer tumor is malignant or benign based on various features. "
    "You can use sliders to input feature values or manually input them. The app will provide predictions using different classifiers."
    "      "
    "      "
    "      "
    )

st.title("Classifier Performance")

# Logistic Regression
st.subheader("Logistic Regression")
train_score_lr, test_score_lr = performance(classifier_lr, x_train_scaled, y_train, x_test_scaled, y_test)
st.write(f"Training Score: {train_score_lr}")
st.write(f"Testing Score: {test_score_lr}")
st.write(f"MAE: {mean_absolute_error(y_test, classifier_lr.predict(x_test_scaled))}")
st.write(f"MSE: {mean_squared_error(y_test, classifier_lr.predict(x_test_scaled))}")
st.write( "      ")
st.write( "      ")

# Random Forest
st.subheader("Random Forest")
train_score_rf, test_score_rf = performance(classifier_rf, x_train_scaled, y_train, x_test_scaled, y_test)
st.write(f"Training Score: {train_score_rf}")
st.write(f"Testing Score: {test_score_rf}")
st.write(f"MAE: {mean_absolute_error(y_test, classifier_rf.predict(x_test_scaled))}")
st.write(f"MSE: {mean_squared_error(y_test, classifier_rf.predict(x_test_scaled))}")
st.write( "      ")
st.write( "      ")

# Decision Tree
st.subheader("Decision Tree")
train_score_dt, test_score_dt = performance(classifier_dt, x_train_scaled, y_train, x_test_scaled, y_test)
st.write(f"Training Score: {train_score_dt}")
st.write(f"Testing Score: {test_score_dt}")
st.write(f"MAE: {mean_absolute_error(y_test, classifier_dt.predict(x_test_scaled))}")
st.write(f"MSE: {mean_squared_error(y_test, classifier_dt.predict(x_test_scaled))}")
st.write( "      ")
st.write( "      ")

# Support Vector Machine (SVM)
st.subheader("Support Vector Machine (SVM)")
train_score_svm, test_score_svm = performance(classifier_svm, x_train_scaled, y_train, x_test_scaled, y_test)
st.write(f"Training Score: {train_score_svm}")
st.write(f"Testing Score: {test_score_svm}")
st.write(f"MAE: {mean_absolute_error(y_test, classifier_svm.predict(x_test_scaled))}")
st.write(f"MSE: {mean_squared_error(y_test, classifier_svm.predict(x_test_scaled))}")
st.write( "      ")
st.write( "      ")

st.sidebar.header("User Input for Prediction")

slider_values = {}

for feature in x_train.columns:
    slider_min = float(x_train[feature].min())
    slider_max = float(x_train[feature].max())
    slider_mean = float(x_train[feature].mean())
    
    slider_values[feature] = st.sidebar.slider(f"{feature} Slider", slider_min, slider_max, slider_mean)
user_input = pd.DataFrame([slider_values])

user_input_text = st.sidebar.text_area("Enter comma-separated values for features (e.g., 13.54, 14.36, 87.46, 566.3, 0.09779, ...):")
if user_input_text:
    user_input_manual = pd.DataFrame([user_input_text.split(',')], columns=x_train.columns)
    user_input_manual = user_input_manual.astype(float)

    for feature in x_train.columns:
        if feature in user_input_manual.columns:
            slider_values[feature] = user_input_manual[feature].values[0]

user_input = pd.DataFrame([slider_values])

user_input_scaled = preprocess_data(user_input, sc_X)

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
