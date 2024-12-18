import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to perform linear regression and display results
def perform_linear_regression(data, x_column, y_column):
    # Select the X and Y values
    X = data[[x_column]].values  # X should be 2D for scikit-learn
    y = data[y_column].values   # y should be 1D
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Train the model on the data
    model.fit(X, y)
    
    # Get the model's coefficients and intercept
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Make predictions based on the model
    y_pred = model.predict(X)
    
    # Display the model details
    st.write(f"Model Coefficient (slope): {slope}")
    st.write(f"Model Intercept: {intercept}")
    
    # Plotting the actual data and the regression line
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Actual Data')  # Scatter plot of the data points
    ax.plot(X, y_pred, color='red', label='Fitted Line')  # Line plot of the regression line
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Linear Regression: {y_column} vs {x_column}")
    ax.legend()
    
    # Display the plot in the Streamlit app
    st.pyplot(fig)

# Streamlit app layout
st.title("Linear Regression on CSV File")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# If a file is uploaded, proceed with the rest
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)
    
    # Show the first few rows of the dataframe
    st.write("Here are the first few rows of your dataset:")
    st.write(data.head())
    
    # Let the user select the X and Y columns
    st.write("Please select the columns for X (independent variable) and Y (dependent variable):")
    
    # Select columns for X and Y from the data
    x_column = st.selectbox("Select X column", data.columns)
    y_column = st.selectbox("Select Y column", data.columns)
    
    # Run the linear regression when both columns are selected
    if x_column and y_column:
        perform_linear_regression(data, x_column, y_column)
