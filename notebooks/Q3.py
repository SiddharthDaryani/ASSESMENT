from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform

# Load the wine dataset
wine_data = load_wine()

# Split data into features (X) and target labels (y)
X = wine_data.data
y = wine_data.target

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    'C': loguniform(1e-5, 100),  # Search for C in a log-uniform range
    'penalty': ['l1', 'l2']   
}

# Create a Logistic Regression model
model = LogisticRegression()

# Create a RandomizedSearchCV object with 100 iterations and 5-fold CV
random_search = RandomizedSearchCV(model, param_dist, n_iter=100, cv=5)

# Fit the model with randomized hyperparameter search on training data
random_search.fit(X_train, y_train)

# Access the best hyperparameters
best_params = random_search.best_params_

print("Best Hyperparameters:", best_params)

# Use the best model to predict on the test set
y_pred = random_search.best_estimator_.predict(X_test)
