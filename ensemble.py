import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,BayesianRidge #, ElasticNet
from sklearn.metrics import  mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn import svm


xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.6, learning_rate = 0.1, n_estimators = 1000,
                max_depth = 5, alpha = 20)
neigh = KNeighborsRegressor(n_neighbors=5)
estimators = [
    ('rf', RandomForestRegressor(n_estimators = 1000)),
    ('xg_reg', xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.6, learning_rate = 0.1, n_estimators = 1000,
                max_depth = 5, alpha = 20)),
    ('neigh', KNeighborsRegressor(n_neighbors=5) ),
    #('clf', BayesianRidge()),
    ('poly',svm.SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)),
    ('rbf',svm.SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.3))
    
    ]
stregr = StackingRegressor(estimators,final_estimator = BayesianRidge(n_iter =1000))


df = pd.read_csv(r'C:\Users\MMiroGranada\BUSINESS INTEGRATION PARTNERS SPA\Decalo Salgado, Laura (Bip) - Grenergy\DataAnalysis\greenergy_dataset_all_features.csv')

# NOTE: Do NOT forget to remove the target from the input data
data = df.set_index('fecha')
data.index = pd.DatetimeIndex(data.index).to_period('D')
data.sort_index(inplace=True)

# anno and acumuladoprec are dropped since they are not good for predictions (they are both counters)
X = data.drop(['Precio medio','anno','acumuladoprec'], axis=1)
y = data['Precio medio']
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('encoder', OneHotEncoder()),
])

_categorical = ['Weekend','Invierno','Verano','Otono','Festivo','mes']
numeric_features = [f for f in X.columns if not f.startswith('cat') and f not in _categorical] # This can be written better
categorical_features = [f for f in X.columns if f.startswith('cat')] + _categorical

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, shuffle = False)

X_cont = X[numeric_features]
X_train_cont,X_test_cont,y_train_cont,y_test_cont = train_test_split(X_cont,y,test_size=0.2, shuffle = False)
X_cat = X[categorical_features]
X_train_cat,X_test_cat,y_train_cat,y_test_cat = train_test_split(X_cat,y,test_size=0.2, shuffle = False)

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ]
)
# =============================================================================
# preprocessor_cat = ColumnTransformer(
#     transformers=[
#         ('categorical', categorical_transformer, categorical_features)
#     ]
# )
# preprocessor_cont = ColumnTransformer(
#     transformers=[
#         ('numeric', numeric_transformer, numeric_features),
#     ]
# )
# 
# =============================================================================
    # Ensemble

pipe = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', stregr)
])
# =============================================================================
# pipe_cat = Pipeline(steps=[
#     ('preprocessing', preprocessor_cat),
#     ('model', stregr)
# ])
# pipe_cont = Pipeline(steps=[
#     ('preprocessing', preprocessor_cont),
#     ('model', stregr)
# ])
# 
# =============================================================================
ENS = pipe.fit(X_train, y_train)
ENS_y_prediction = pipe.predict(X_test)

# predicting the accuracy score
ENS_score = np.sqrt(mean_squared_error(y_test,ENS_y_prediction))

df_predictions = pd.DataFrame({
    'real': y_test,
    'predicted_ENS': ENS_y_prediction
})
#df_predictions.to_excel(r'pred.xlsx', index = True)
df_predictions.plot(figsize=(12,8), title='Comparación de valor predicho y real con variables categ')

ENS_score
#


