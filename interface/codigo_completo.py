import pandas as pd
import numpy as np
import holidays

import pickle

oil = 3
gas = 3
eua = 3
cer = 3

vel = 3
sol = 1
prec = 3
temp = 2

baseline = pd.read_csv(r'C:\Users\MMiroGranada\BUSINESS INTEGRATION PARTNERS SPA\Decalo Salgado, Laura (Bip) - Grenergy\interface\Data\baseline.csv',
                 sep=',', decimal='.')
baseline = baseline.iloc[:, 0:10]

maxx = [max(baseline.gas), max(baseline.petroleo), max(baseline.ton_CO2), max(baseline.CER_CO2),
        max(baseline.prec), max(baseline.sol), max(baseline.tmed), max(baseline.velmedia)]

meann = [np.mean(baseline.gas), np.mean(baseline.petroleo), np.mean(baseline.ton_CO2),
         np.mean(baseline.CER_CO2), np.mean(baseline.prec), np.mean(baseline.sol),
         np.mean(baseline.tmed), np.mean(baseline.velmedia)]

factor = {'petroleo': (maxx[0]-meann[0])*0.75/meann[0],
        'gas':        (maxx[1]-meann[1])*0.75/meann[1],
        'ton_CO2':    (maxx[2]-meann[2])*0.75/meann[2],
        'CER_CO2':  (maxx[3]-meann[3])*0.75/meann[3],
        'velmedia': (maxx[4]-meann[4])*0.75/meann[4],
        'sol':      (maxx[5]-meann[5])*0.75/meann[5],
        'prec':     (maxx[6]-meann[6])*0.75/meann[6],
        'tmed':     (maxx[7]-meann[7])*0.75/meann[7] }

ints = {
        'gas':      gas,
        'ton_CO2':  eua,
        'CER_CO2':  cer,
        'petroleo': oil,
        'prec':     prec,
        'sol':      sol,
        'tmed':     temp,
        'velmedia': vel }

fecha = pd.date_range(start='1/1/2020', end='31/12/2020', freq='D')
df = pd.DataFrame(fecha,columns = ['fecha'])
df['fecha'] = pd.to_datetime(df['fecha']).dt.date


for key,value in ints.items():
    if value == 1:
        df[key] = baseline[key]-factor[key]*baseline[key]
    elif value == 3:
        df[key] = baseline[key]+factor[key]*baseline[key]
    else:
        df[key] = baseline[key]
        

es_holidays = holidays.ES()

weekend = []
invierno = []
verano = []
otono = []
holiday = []

for i in range(len(df)):
    aux_week_day = df.loc[i,'fecha'].weekday()
    
    if aux_week_day in [0, 1, 2, 3, 4]:
         aux_day = 0
    else:
         aux_day = 1
    
    if df.loc[i,'fecha'].month in [1, 2, 12]:
        aux_invierno = 1
    else:
        aux_invierno = 0
    
    if df.loc[i,'fecha'].month in [9, 10, 11]:
        aux_otono = 1
    else:
        aux_otono = 0
    
    if df.loc[i,'fecha'].month in [6, 7, 8]:
        aux_verano = 1
    else:
        aux_verano = 0
        
    if df.loc[i,'fecha'] in es_holidays:
        aux_holiday = 1
    else:
        aux_holiday = 0

    weekend.append(aux_day)
    invierno.append(aux_invierno)
    verano.append(aux_verano)
    otono.append(aux_otono)
    holiday.append(aux_holiday)

df['Weekend'] = weekend
df['Invierno'] = invierno
df['Verano'] = verano
df['Otono'] = otono
df['Festivo'] = holiday
df['mes'] = df.fecha.apply(lambda x: x.month)

prec_q1 = df[['mes', 'prec']].groupby(['mes']).quantile(.25).rename(columns={'prec':'prec_q1'})
prec_q2 = df[['mes', 'prec']].groupby(['mes']).quantile(.5).rename(columns={'prec':'prec_q2'})
prec_q3 = df[['mes', 'prec']].groupby(['mes']).quantile(.75).rename(columns={'prec':'prec_q3'})

sol_q1 = df[['mes', 'sol']].groupby(['mes']).quantile(.25).rename(columns={'sol':'sol_q1'})
sol_q2 = df[['mes', 'sol']].groupby(['mes']).quantile(.5).rename(columns={'sol':'sol_q2'})
sol_q3 = df[['mes', 'sol']].groupby(['mes']).quantile(.75).rename(columns={'sol':'sol_q3'})

tmed_q1 = df[['mes', 'tmed']].groupby(['mes']).quantile(.25).rename(columns={'tmed':'tmed_q1'})
tmed_q2 = df[['mes', 'tmed']].groupby(['mes']).quantile(.5).rename(columns={'tmed':'tmed_q2'})
tmed_q3 = df[['mes', 'tmed']].groupby(['mes']).quantile(.75).rename(columns={'tmed':'tmed_q3'})

velmedia_q1 = df[['mes', 'velmedia']].groupby(['mes']).quantile(.25).rename(columns={'velmedia':'velmedia_q1'})
velmedia_q2 = df[['mes', 'velmedia']].groupby(['mes']).quantile(.5).rename(columns={'velmedia':'velmedia_q2'})
velmedia_q3 = df[['mes', 'velmedia']].groupby(['mes']).quantile(.75).rename(columns={'velmedia':'velmedia_q3'})

petroleo_q1 = df[['mes', 'petroleo']].groupby(['mes']).quantile(.25).rename(columns={'petroleo':'petroleo_q1'})
petroleo_q2 = df[['mes', 'petroleo']].groupby(['mes']).quantile(.5).rename(columns={'petroleo':'petroleo_q2'})
petroleo_q3 = df[['mes', 'petroleo']].groupby(['mes']).quantile(.75).rename(columns={'petroleo':'petroleo_q3'})

gas_q1 = df[['mes', 'gas']].groupby(['mes']).quantile(.25).rename(columns={'gas':'gas_q1'})
gas_q2 = df[['mes', 'gas']].groupby(['mes']).quantile(.5).rename(columns={'gas':'gas_q2'})
gas_q3 = df[['mes', 'gas']].groupby(['mes']).quantile(.75).rename(columns={'gas':'gas_q3'})

ton_CO2_q1 = df[['mes', 'ton_CO2']].groupby(['mes']).quantile(.25).rename(columns={'ton_CO2':'ton_CO2_q1'})
ton_CO2_q2 = df[['mes', 'ton_CO2']].groupby(['mes']).quantile(.5).rename(columns={'ton_CO2':'ton_CO2_q2'})
ton_CO2_q3 = df[['mes', 'ton_CO2']].groupby(['mes']).quantile(.75).rename(columns={'ton_CO2':'ton_CO2_q3'})

CER_CO2_q1 = df[['mes', 'CER_CO2']].groupby(['mes']).quantile(.25).rename(columns={'CER_CO2':'CER_CO2_q1'})
CER_CO2_q2 = df[['mes', 'CER_CO2']].groupby(['mes']).quantile(.5).rename(columns={'CER_CO2':'CER_CO2_q2'})
CER_CO2_q3 = df[['mes', 'CER_CO2']].groupby(['mes']).quantile(.75).rename(columns={'CER_CO2':'CER_CO2_q3'})

cuantiles = pd.concat([prec_q1, prec_q2, prec_q3, 
                      sol_q1, sol_q2, sol_q3,
                      tmed_q1, tmed_q2, tmed_q3,
                      velmedia_q1, velmedia_q2, velmedia_q3,
                      petroleo_q1,petroleo_q2,petroleo_q3,
                      gas_q1, gas_q2, gas_q3,
                      ton_CO2_q1,ton_CO2_q2,ton_CO2_q3,
                      CER_CO2_q1,CER_CO2_q2,CER_CO2_q3
                      ], axis=1).reset_index()


cat_prec = []

cat_sol = []
cat_tmed = []
cat_velmedia = []
cat_petroleo = []
cat_gas = []
cat_ton_CO2 = []
cat_CER_CO2 = []

for i in range(len(df)):
    
    aux_mes = df.loc[i, 'mes']
    
    if df.loc[i, 'prec'] < cuantiles.loc[aux_mes-1, 'prec_q1']:
        aux_cat_prec = 1
    elif df.loc[i, 'prec'] > cuantiles.loc[aux_mes-1, 'prec_q3']:
        aux_cat_prec = 3
    else:
        aux_cat_prec = 2
        
    cat_prec.append(aux_cat_prec)
    
    if df.loc[i, 'sol'] < cuantiles.loc[aux_mes-1, 'sol_q1']:
        aux_cat_sol = 1
    elif df.loc[i, 'sol'] > cuantiles.loc[aux_mes-1, 'sol_q3']:
        aux_cat_sol = 3
    else:
        aux_cat_sol = 2
        
    cat_sol.append(aux_cat_sol)
    
    if df.loc[i, 'tmed'] < cuantiles.loc[aux_mes-1, 'tmed_q1']:
        aux_cat_tmed = 1
    elif df.loc[i, 'tmed'] > cuantiles.loc[aux_mes-1, 'tmed_q3']:
        aux_cat_tmed = 3
    else:
        aux_cat_tmed = 2
        
    cat_tmed.append(aux_cat_tmed)
    
    if df.loc[i, 'velmedia'] < cuantiles.loc[aux_mes-1, 'velmedia_q1']:
        aux_cat_velmedia = 1
    elif df.loc[i, 'velmedia'] > cuantiles.loc[aux_mes-1, 'velmedia_q3']:
        aux_cat_velmedia = 3
    else:
        aux_cat_velmedia = 2
        
    cat_velmedia.append(aux_cat_velmedia)
    
    if df.loc[i, 'petroleo'] < cuantiles.loc[aux_mes-1, 'petroleo_q1']:
        aux_cat_petroleo = 1
    elif df.loc[i, 'petroleo'] > cuantiles.loc[aux_mes-1, 'petroleo_q3']:
        aux_cat_petroleo = 3
    else:
        aux_cat_petroleo = 2
        
    cat_petroleo.append(aux_cat_petroleo)
    
    if df.loc[i, 'gas'] < cuantiles.loc[aux_mes-1, 'gas_q1']:
        aux_cat_gas = 1
    elif df.loc[i, 'gas'] > cuantiles.loc[aux_mes-1, 'gas_q3']:
        aux_cat_gas = 3
    else:
        aux_cat_gas = 2
        
    cat_gas.append(aux_cat_gas)
    
    if df.loc[i, 'ton_CO2'] < cuantiles.loc[aux_mes-1, 'ton_CO2_q1']:
        aux_cat_ton_CO2 = 1
    elif df.loc[i, 'ton_CO2'] > cuantiles.loc[aux_mes-1, 'ton_CO2_q3']:
        aux_cat_ton_CO2 = 3
    else:
        aux_cat_ton_CO2 = 2
        
    cat_ton_CO2.append(aux_cat_ton_CO2)
    
    if df.loc[i, 'CER_CO2'] < cuantiles.loc[aux_mes-1, 'CER_CO2_q1']:
        aux_cat_CER_CO2 = 1
    elif df.loc[i, 'CER_CO2'] > cuantiles.loc[aux_mes-1, 'CER_CO2_q3']:
        aux_cat_CER_CO2 = 3
    else:
        aux_cat_CER_CO2 = 2
        
    cat_CER_CO2.append(aux_cat_CER_CO2)
    
df['cat_prec'] = cat_prec
df['cat_sol'] = cat_sol
df['cat_tmed'] = cat_tmed
df['cat_velmedia'] = cat_velmedia
df['cat_petroleo'] = cat_petroleo
df['cat_gas'] = cat_gas
df['cat_ton_CO2'] = cat_ton_CO2
df['cat_CER_CO2'] = cat_CER_CO2

del(cat_prec,cat_sol,cat_tmed,cat_velmedia,cat_petroleo,cat_gas,cat_ton_CO2,cat_CER_CO2,
    aux_cat_prec,aux_cat_sol,aux_cat_tmed,aux_cat_velmedia,aux_cat_petroleo,aux_cat_gas,aux_cat_ton_CO2,aux_cat_CER_CO2,
    prec_q1, prec_q2, prec_q3,sol_q1, sol_q2, sol_q3, tmed_q1, tmed_q2, tmed_q3,
    velmedia_q1, velmedia_q2, velmedia_q3,petroleo_q1,petroleo_q2,petroleo_q3,
    gas_q1, gas_q2, gas_q3,ton_CO2_q1,ton_CO2_q2,ton_CO2_q3,CER_CO2_q1,CER_CO2_q2,CER_CO2_q3
    #,otono,verano,weekend,invierno,cuantiles,aux_day,aux_holiday,aux_invierno,aux_mes,aux_otono,aux_verano,aux_week_day
    )

data = df.set_index('fecha')
data.index = pd.DatetimeIndex(data.index).to_period('D')
data.sort_index(inplace=True)

# =============================================================================
# Tiene que quedar un archivo con una fecha de indice y todas las columnas
# =============================================================================

    # Linear regression

loaded_model = pickle.load(open(r'Data/Modelos entrenados/linearregression.sav', 'rb'))
LR_y_prediction = loaded_model.predict(data)

   # RF regression

loaded_model = pickle.load(open(r'Data/Modelos entrenados/randomforest.sav', 'rb'))
RF_y_prediction = loaded_model.predict(data)

    # XGBOOST

loaded_model = pickle.load(open(r'Data/Modelos entrenados/xgb.sav', 'rb'))
XGB_y_prediction = loaded_model.predict(data)

    # SVM poly

loaded_model = pickle.load(open(r'Data/Modelos entrenados/svm.sav', 'rb'))
SVM_POLY_y_prediction = loaded_model.predict(data)

    # ENSEMBLE
    
weights = pickle.load(open(r'Data/Modelos entrenados/weights.pkl', 'rb'))
weights['weight_lr']

ensemble = RF_y_prediction*weights['weight_rf']+LR_y_prediction*weights['weight_lr']+XGB_y_prediction*weights['weight_xgb']+SVM_POLY_y_prediction*weights['weight_svm']

    # Se crea el dataset con los valores predichos
result = {
       'linear_regression': LR_y_prediction ,
       'Random_Forest': RF_y_prediction,
        'XGB': XGB_y_prediction,
       'SVM_POLY': SVM_POLY_y_prediction,
        'ensemble': ensemble
       }

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
    
        preds['price_mean'] = preds['precio'] * preds['mean']
        preds['price_low'] = preds['precio'] * preds['low_bound']
        preds['price_high'] = preds['precio'] * preds['high_bound']
    
        return preds[['fecha','Hora', 'price_low', 'price_mean', 'price_high']]

price_factors = pd.read_csv(r"C:\Users\MMiroGranada\BUSINESS INTEGRATION PARTNERS SPA\Decalo Salgado, Laura (Bip) - Grenergy\interface\Data\hourly_price_factors.csv", delimiter=";", decimal=',', index_col=0)

for  i in result:
    
    data['precio'] = result[i]

    daily_predictions = data[['precio']]
    daily_predictions = daily_predictions.reset_index()
    daily_predictions.fecha  = daily_predictions.fecha.astype(str)
    daily_predictions.fecha = pd.to_datetime(daily_predictions.fecha)
    
        
    
    daily_predictions.reset_index(drop=True, inplace=True)
    
    daily_predictions.head()
    
    hourly_prices = convert_daily_to_hourly_predictions(daily_predictions, price_factors)
    out_path= r"\Data\resultados"+'_'+i+'.csv'
    hourly_prices.to_csv(out_path, sep=',', decimal='.')
    del data["precio"]