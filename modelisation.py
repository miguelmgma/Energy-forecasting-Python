import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv(r'C:\Users\MMiroGranada\BUSINESS INTEGRATION PARTNERS SPA\Decalo Salgado, Laura (Bip) - Grenergy\DataAnalysis\greenergy_dataset_all_features.csv')

# NOTE: Do NOT forget to remove the target from the input data
data = df.set_index('fecha')
data.index = pd.DatetimeIndex(data.index).to_period('D')
data.sort_index(inplace=True)

# anno and acumuladoprec are dropped since they are not good for predictions (they are both counters)
X = data.drop(['Precio medio','anno','acumuladoprec','cat_acumuladoprec'], axis=1)
y = data['Precio medio']
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, shuffle = False)


    # Linear regression

loaded_model = pickle.load(open(r'Modelos entrenados/linearregression.sav', 'rb'))
LR_y_prediction = loaded_model.predict(X_test)

   # RF regression

loaded_model = pickle.load(open(r'Modelos entrenados/randomforest.sav', 'rb'))
RF_y_prediction = loaded_model.predict(X_test)

    # XGBOOST

loaded_model = pickle.load(open(r'Modelos entrenados/xgb.sav', 'rb'))
XGB_y_prediction = loaded_model.predict(X_test)

    # SVM poly

loaded_model = pickle.load(open(r'Modelos entrenados/svm.sav', 'rb'))
SVM_POLY_y_prediction = loaded_model.predict(X_test)

    # ENSEMBLE
    
w = (np.sqrt(mean_squared_error(y_test,RF_y_prediction))**-1+np.sqrt(mean_squared_error(y_test,LR_y_prediction))**-1+
    np.sqrt(mean_squared_error(y_test,XGB_y_prediction))**-1+np.sqrt(mean_squared_error(y_test,SVM_POLY_y_prediction))**-1)

w_rf = (1/np.sqrt(mean_squared_error(y_test,RF_y_prediction)))/w
w_lr = (1/np.sqrt(mean_squared_error(y_test,LR_y_prediction)))/w
w_xgb = (1/np.sqrt(mean_squared_error(y_test,XGB_y_prediction)))/w
w_svm = (1/np.sqrt(mean_squared_error(y_test,SVM_POLY_y_prediction)))/w
ensemble = RF_y_prediction*w_rf+LR_y_prediction*w_lr+XGB_y_prediction*w_xgb+SVM_POLY_y_prediction*w_svm
weights = {'weight_rf':w_rf,'weight_lr':w_lr,'weight_xgb':w_xgb,'weight_svm': w_svm}
filename_w = 'weights.pkl'
with open(filename_w, 'wb') as file:
    pickle.dump(weights, file)

res1 = pd.DataFrame({'real':y_test ,
       'linear_regression': LR_y_prediction ,
       'Random_Forest': RF_y_prediction,
       # 'XGB': XGB_y_prediction,
       'SVM_POLY': SVM_POLY_y_prediction,
       # 'ensemble': ensemble
       })
res2 = pd.DataFrame({'real':y_test ,
       # 'linear_regression': LR_y_prediction ,
       # 'Random_Forest': RF_y_prediction,
       'XGB': XGB_y_prediction,
       # 'SVM_POLY': SVM_POLY_y_prediction,
       'ensemble': ensemble
       })
    
res1.plot( figsize=(40,20))
res2.plot( figsize=(40,20))
    # Se crea el dataset con los valores predichos

daily_predictions = pd.DataFrame(
        {
         'real': y_test,
         'ensemble': ensemble,
         })

daily_predictions = daily_predictions.reset_index()
daily_predictions.fecha  = daily_predictions.fecha.astype(str)
daily_predictions.fecha = pd.to_datetime(daily_predictions.fecha)
np.sqrt(mean_squared_error(y_test,ensemble))
np.mean()
    # De diario a horario
    
def type_of_day(date: pd.Timestamp) -> str:
    return 'Laborable' if date.weekday() < 5 else 'No laborable' 

def extract_month(date: pd.Timestamp) -> int:
    return date.month

def convert_daily_to_hourly_predictions(preds: pd.DataFrame, price_factors: pd.DataFrame) -> pd.DataFrame:
    # Marking predictions as working or not-working days
    preds['tipo_dia'] = preds['fecha'].apply(type_of_day)
    preds['Mes'] = preds['fecha'].apply(extract_month)

    # Merge information of price factors and predictions
    preds = pd.merge(preds, price_factors, on=['Mes','tipo_dia'])

    preds['price_mean'] = preds['ensemble'] * preds['mean']
    preds['price_low'] = preds['ensemble'] * preds['low_bound']
    preds['price_high'] = preds['ensemble'] * preds['high_bound']

    return preds[['fecha','Hora', 'price_low', 'price_mean', 'price_high']]

price_factors = pd.read_csv(r"Data\hourly_price_factors.csv", delimiter=";", decimal=',', index_col=0)

daily_predictions.reset_index(drop=False, inplace=True)

daily_predictions.head()

hourly_prices = convert_daily_to_hourly_predictions(daily_predictions, price_factors)
#hourly_prices.to_csv(r"pred.csv", sep=',', decimal='.')