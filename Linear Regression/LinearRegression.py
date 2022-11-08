# Import the libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

# I will use MSE for evaluation
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# Methods to help tp evaluate the models.
def plot_errors(lambdas, train_errors, test_errors, title):
    plt.figure(figsize = (16, 9))
    plt.plot(lambdas, train_errors, label = "Train error")
    plt.plot(lambdas, test_errors, label = "Test error")
    plt.xlabel("$\\lambda$", fontsize = 15)
    plt.ylabel("MSE", fontsize = 15)
    plt.title(title, fontsize = 20)
    plt.legend(fontsize = 15)
    plt.show()

def evaluate_model(Model, lambdas):
    
    # We will store the error on the training set, for using each different lambda
    training_errors = []

    # And the error on the testing set
    testing_errors = []
    
    for lambdaValue in lambdas:
        
        # Model will be Lasso, Ridge and ElasticNet
        model = Model(alpha = lambdaValue, max_iter = 1000)
        model.fit(X_train, y_train)

        training_predictions = model.predict(X_train)
        training_mse = mean_squared_error(y_train, training_predictions)
        training_errors.append(training_mse)

        testing_predictions = model.predict(X_test)
        testing_mse = mean_squared_error(y_test, testing_predictions)
        testing_errors.append(testing_mse)
        
    return training_errors, testing_errors

# Load the dataset
dataSet = pd.read_csv("Salary_Data.csv")

X = dataSet[["YearsExperience"]]
y = dataSet[["Salary"]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Create the model without regularization
model = LinearRegression()
model.fit(X_train, y_train)

plt.xlabel("Years of experience", fontsize = 20)
plt.ylabel("Salary", fontsize = 20)
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, model.predict(X_train), color = "green")

#Calculate the error
training_mseL = mean_squared_error(y_train, model.predict(X_train))
test_mseL = mean_squared_error(y_test, model.predict(X_test))
plt.figtext(0.15, 0.83, f'Train error: = {training_mseL}, Test error: = {test_mseL}')
plt.show()

# Regularization
# Evaluate the models

# Lasso
lambdas = np.arange(0.2, 10, step = 0.1)
lasso_train, lasso_test = evaluate_model(Lasso, lambdas)
plot_errors(lambdas, lasso_train, lasso_test, "Lasso")

# Ridge
ridge_train, ridge_test = evaluate_model(Ridge, lambdas)
plot_errors(lambdas, ridge_train, ridge_test, "Ridge")

# Elastic Net
elastic_train, elastic_test = evaluate_model(ElasticNet, lambdas)
plot_errors(lambdas, elastic_train, elastic_test, "Elastic Net")



