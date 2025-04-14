# Import necessary libraries
import pandas as pd                      # For handling and processing data (CSV, etc.)
import numpy as np                       # For numerical operations and array manipulations
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler       # For feature scaling
from sklearn.linear_model import LogisticRegression     # Logistic Regression model
from sklearn.metrics import accuracy_score              # For evaluating model accuracy

# Create an instance of StandardScaler for feature normalization
sc = StandardScaler()

# Load dataset from CSV file
dataset = pd.read_csv('data.csv')

# Extract features (all columns except the last)
x = dataset.iloc[:, :-1].values
print(x)  # Optional: print feature values

# Extract target variable (last column)
y = dataset.iloc[:, -1].values
print(y)  # Optional: print labels

# Split dataset into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Apply feature scaling: fit on training data, transform both training and test data
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict on the test data
y_pred = model.predict(x_test)

# Print predicted labels
print(y_pred)

# Show a side-by-side comparison of predicted vs actual values
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

# Get user input for new prediction
age = int(input("Enter the Age:"))
sal = int(input("Enter the Salary:"))
x = [[age, sal]]  # Wrap input into a 2D list (since model expects array-like input)

# Scale input using the same scaler fitted on training data
result = model.predict(sc.transform(x))

# Print prediction result
print(result)
if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")

# Evaluate and print model accuracy
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred) * 100))
