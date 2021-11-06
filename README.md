# Programming Homework #1
- Auto ML for classification

## findBestOptions
- Goal: This function will try combinations of the various models automatically.
- This function finds the best model among DecisionTreeClassifier, LogisticRegression, and SVC. And find the best parameter using cross validation.

### Parameters
- `X`: training dataset
- `y`: target
- `scalers`: StandardScaler, RubustScaler, MinMaxScaler, MaxAbsScaler
- `models`
  - DecisionTreeClassifier(criterion=criterion)
    - criterion = ["gini", "entropy"]
  - LogisticRegression(solver=solver)
    - solver = ["newton-cg", "lbfgs", "liblinear"]
  - SVC(kernel=kernel, gamma=gamma)
    - kernel = ["linear", "poly", "rbf", "sigmoid"]
    - gamma = [0.001, 0.01, 0.1, 1, 10]

### Returns
- best_params
  -  best_scaler
  -  best_model
  -  best_cv_k
  -  maxScore
