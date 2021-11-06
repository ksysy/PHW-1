import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt

import seaborn as sns

#classification
criterion=["gini", "entropy"]
solver=["newton-cg", "lbfgs", "liblinear"]
kernel=["linear", "poly", "rbf", "sigmoid"]
gamma=[0.001, 0.01, 0.1, 1, 10]

models=[]
for c in criterion:
    models.append(DecisionTreeClassifier(criterion=c))
for s in solver:
    models.append(LogisticRegression(solver=s, max_iter=500))
for k in kernel:
    for g in gamma:
        models.append(SVC(kernel=k, gamma=g))


#Function for find best options
def findBestOptions(
    X, y,
    scalers=[StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()],
    models=models,
    cv_k=range(2,7)):

    maxScore=-1.0
    best_scaler=None
    best_model=None

    #find best scaler
    for s in range(0, len(scalers)):
        X=scalers[s].fit_transform(X)
    #find best model
        for m in range(0, len(models)):
            #find best cross_val_k cv
            for k in cv_k:
                kfold=KFold(n_splits=k, shuffle=True)
                score_result=cross_val_score(models[m], X, y, cv=kfold)
                #find best score
                if maxScore<score_result.mean():
                    maxScore=score_result.mean(),
                    best_scaler=scalers[s]
                    best_model=models[m]
                    best_cv_k=k

    return{
        'best_params_':{
            'best_scaler': best_scaler,
            'best_model': best_model,
            'best_cv_k': best_cv_k,
            'maxScore': maxScore
        }
    }



#Source Code
#wisconsin Data load
df = pd.read_csv("./breast-cancer-wisconsin.data",
names=['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
    'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'
])

#Data Preprocessing
df = df.drop("Sample code number", axis=1)
df = df.replace('?', np.nan)
df = df.dropna(axis=0)
df['Bare Nuclei']=pd.to_numeric(df['Bare Nuclei'])
X = df.drop("Class", axis=1)
y = df.Class
df['Class'] = df['Class'].replace({2:0, 4:1})

#Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


results=findBestOptions(X, y)
print(results)

result=results['best_params_']
best_model=result['best_model']
best_scaler=result['best_scaler']

X = best_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = best_model.fit(X_train, y_train)
print("Score:", model.score(X_test, y_test))


# #visualization
sns.set_style("whitegrid")
sns.pairplot(df, hue="Class")
plt.show()
