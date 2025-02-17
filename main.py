
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

if __name__ == "__main__":
    # Load the data
    data_root = "https://github.com/ageron/data/raw/main/"
    lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
    X = lifesat[["GDP per capita (USD)"]].values
    y = lifesat[["Life satisfaction"]].values

    # Calculate the bias term
    X_b = np.c_[np.ones((len(X), 1)), X] # add x0 = 1 to each instance 
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # Normal Equation derived from the cost function (MSE)

    model = LinearRegression()
    model.fit(X, y)

    print("Intercept and Coefficients from Normal Equation: ", theta)
    print("Intercept and Coefficients from Scikit-Learn: ", model.intercept_, model.coef_)

    # Make a prediction for Cyprus
    X_new = [[22587]]  # Cyprus' GDP per capita
    print("Prediction for Cyprus using Normal Equation: ", X_new[0][0] * theta[1] + theta[0])
    print("Prediction for Cyprus using Scikit-Learn: ", model.predict(X_new))

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y)
    plt.xlabel("GDP per capita (USD)")
    plt.ylabel("Life satisfaction")
    plt.title("Life Satisfaction vs GDP per Capita")
    plt.plot(X, X_b.dot(theta), "r-")

    plt.show()
