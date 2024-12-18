import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to perform linear regression
def perform_linear_regression(file_path):
    # Load the CSV data into a DataFrame
    data = pd.read_csv(file_path)
    
    # Display the available columns in the dataset
    print("\nColumns available in the CSV file:")
    print(data.columns.tolist())
    
    # Get user input for the X and Y columns
    x_column = input("\nEnter the name of the X column (independent variable): ").strip()
    y_column = input("Enter the name of the Y column (dependent variable): ").strip()
    
    # Check if the selected columns exist in the dataset
    if x_column not in data.columns or y_column not in data.columns:
        print("Error: Invalid column names. Please check the column names and try again.")
        return
    
    # Select the X and Y values
    X = data[[x_column]].values  # X should be 2D for scikit-learn
    y = data[y_column].values   # y should be 1D
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Train the model on the data
    model.fit(X, y)
    
    # Print the model's coefficients and intercept
    print(f"\nModel Coefficient (slope): {model.coef_[0]}")
    print(f"Model Intercept: {model.intercept_}")
    
    # Make predictions based on the model
    y_pred = model.predict(X)
    
    # Plot the actual data and the regression line
    plt.scatter(X, y, color='blue', label='Actual Data')  # Scatter plot of the data points
    plt.plot(X, y_pred, color='red', label='Fitted Line')  # Line plot of the regression line
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"Linear Regression: {y_column} vs {x_column}")
    plt.legend()
    plt.show()

# Main function to execute the program
if __name__ == "__main__":
    # Ask for the file path
    file_path = input("Enter the path to your CSV file: ").strip()
    
    # Perform linear regression
    perform_linear_regression(file_path)
