import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Function to fetch data from the Random User Generator API
def fetch_data(num_samples=300):
    url = f'https://randomuser.me/api/?results={num_samples}'
    response = requests.get(url)
    data = response.json()
    usernames = [user['login']['username'] for user in data['results']]
    ages = [user['dob']['age'] for user in data['results']]
    return usernames, ages


usernames, ages = fetch_data()

# Use the length of usernames as the feature
X = np.array([len(username) for username in usernames]).reshape(-1, 1)
y = np.array(ages)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Simple Linear Regression on Random User Generator API Data')
plt.xlabel('Number of Characters in Username')
plt.ylabel('Age')
plt.show()
