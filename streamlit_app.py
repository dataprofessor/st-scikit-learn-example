import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_data():
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
  # Separate data as X and y
  X = df.drop('logS', axis=1)
  y = df['logS']
  return X, y

# Preprocess data
@st.cache_data
def preprocess_data():
  # Data splitting
  return X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_data
def model_building():
  # Model building
  rf = RandomForestRegressor(max_features=4, n_estimators=100)
  rf.fit(X_train, y_train)
  # Apply trained model to make predictions
  y_train_pred = rf.predict(X_train)
  y_test_pred = rf.predict(X_test)
  return y_train_pred, y_test_pred

@st.cache_data
def model_performance():
  # Evaluate model performance
  train_r2 = r2_score(y_train, y_train_pred)
  test_r2 = r2_score(y_test, y_test_pred)
  train_mse = mean_squared_error(y_train, y_train_pred)
  test_mse = mean_squared_error(y_test, y_test_pred)
  return train_r2, test_r2, train_mse, test_mse

X, y = load_data()
#X_train, X_test, y_train, y_test = preprocess_data()
#y_train_pred, y_test_pred = model_building()
#train_r2, test_r2, train_mse, test_mse = model_performance()

st.title('🤖 Scikit-learn - A minimum working example')

st.write('Actual y')
#st.write(y_train)

st.write('Predicted y')
#st.write(y_train_pred)

st.write(X)

st.write(y)
st.write('Train $R^2$', train_r2)
st.write('Test $R^2$', test_r2)
