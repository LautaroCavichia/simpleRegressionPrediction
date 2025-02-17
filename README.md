# Linear Regression with Normal Equation and Scikit-Learn

This project demonstrates how to perform **Linear Regression** using both the **Normal Equation** and **Scikit-Learn's LinearRegression model**. The dataset used is "Life Satisfaction vs GDP per Capita" from Aurélien Géron's *Hands-On Machine Learning*.

## Description

The script performs the following steps:

1. Loads the dataset containing GDP per capita and life satisfaction scores.

2. Computes the optimal parameters (**intercept and slope**) using the **Normal Equation**, which is derived from minimizing the Mean Squared Error (MSE):

   $$
   \theta = (X^T X)^{-1} X^T y
   $$

3. Trains a linear regression model using **Scikit-Learn**.

4. Compares the parameters obtained from both methods.

5. Predicts the **life satisfaction** for Cyprus based on its GDP per capita.

6. Plots the data along with the regression line computed from the Normal Equation.

## Code Breakdown

### Loading the Dataset

The dataset is fetched from an online repository and contains GDP per capita and life satisfaction values:

```python
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values
```

### Computing Parameters using the Normal Equation

We extend the feature matrix \(X\) by adding a bias term (column of ones) to account for the intercept:

```python
X_b = np.c_[np.ones((len(X), 1)), X]  # Add x0 = 1 to each instance
```

Then, we compute \(\theta\) using the **Normal Equation**:

```python
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```

This formula comes from the derivative of the MSE cost function.

### Training a Model with Scikit-Learn

To validate our results, we also train a model using Scikit-Learn’s built-in **LinearRegression**:

```python
model = LinearRegression()
model.fit(X, y)
```

### Making Predictions for Cyprus

Using both methods, we predict life satisfaction for a GDP per capita of **22,587 USD**:

```python
X_new = [[22587]]
print("Prediction using Normal Equation:", X_new[0][0] * theta[1] + theta[0])
print("Prediction using Scikit-Learn:", model.predict(X_new))
```

### Plotting the Results

We visualize the data points along with the fitted regression line:

```python
plt.scatter(X, y)
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.title("Life Satisfaction vs GDP per Capita")
plt.plot(X, X_b.dot(theta), "r-")  # Regression line
plt.show()
```

## License

This project follows the concepts from *Hands-On Machine Learning* by Aurélien Géron and is for educational purposes.

