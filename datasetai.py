import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data into a pandas dataframe
df = pd.read_csv('data.csv')

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

print("\nX_train: \n", X_train)
print("\nX_test: \n", X_test)
print("\ny_train: \n", y_train)
print("\ny_test: \n", y_test)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training set
model.fit(X_train, y_train)

# Evaluate the model using the test set
score = model.score(X_test, y_test)
print(f'Model score: {score:.2f}')

pmf = pd.read_csv('predictme.csv')
# Use the model to make predictions on new data
predictions = model.predict(pmf)

print("\nPredictions: \n", predictions)
