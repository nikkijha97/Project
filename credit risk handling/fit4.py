import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle 

data = pd.read_excel("data/default of credit card clients.xls")
target = data['default payment next month'].values
data = data.drop(['default payment next month'], axis = 1)

for variable in data.columns[6:12]:
    data[variable + "_no_card_use"] = np.where(data[variable] == -2, 1, 0)
    data[variable + "_payed_off"] = np.where(data[variable] == -1, 1, 0)
    data[variable] = np.where(data[variable] < 0, 0, data[variable])

categorical_features=['SEX', 'EDUCATION', 'MARRIAGE']
preprocessor = Pipeline([
  # Step 1: Apply one-hot encoding to categorical features
        ('categorical', ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 categorical_features)
            ],
            remainder='passthrough'  # Keep numerical features as is
        )),
        # Step 2: Scale all features (both encoded and numerical)
        ('scaler', StandardScaler())
    ])

preprocessor.fit(data)
#filename = "preprocessor.pkl"
#pickle.dump(preprocessor, open(filename, 'wb'))
pre_data = preprocessor.transform(data)
def optimize_classifier(train_data, train_target):
    """
    Perform grid search to find the best classifier parameters
    """
    classifier = GradientBoostingClassifier(random_state=1000)
    optimizer = GridSearchCV(
        classifier,
        param_grid={
            'learning_rate': [0.05],
            'n_estimators': [250],
            'max_depth': [5],
            'subsample': [0.6],
            'max_features': ['sqrt', 'log2']
        },
        cv=10,
        scoring='roc_auc',
        n_jobs=4,
        verbose=1
    )
    
    optimizer.fit(train_data, train_target)
    return optimizer.best_estimator_
  
train_data, test_data,train_target,test_target = train_test_split(
    pre_data,
    target,
    train_size=0.8,
    random_state=1000,
    shuffle=True)
best_estimator = optimize_classifier(train_data, train_target)

fitted_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Fitted preprocessor
    ('classifier', best_estimator)  # Fitted model
])
filename = "fitted_pipeline.pkl"
pickle.dump(fitted_pipeline, open(filename, 'wb'))


    
    
  
